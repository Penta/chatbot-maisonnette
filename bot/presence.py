"""Statut Discord du bot : l'anime qu'il regarde, la musique qu'il écoute, le jeu auquel il joue…

Chaque statut est tiré au sort selon PRESENCE_WEIGHTS, avec sa propre durée (quelques épisodes,
quelques morceaux, une session de jeu…) ; à la fin, un nouveau statut est tiré. Pendant la nuit
(PRESENCE_NIGHT), le bot est absent.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

import discord

from .presence_sources import Game, PresenceSources, SourceError, Track

if TYPE_CHECKING:
    from discord.ext import commands

    from .claude_client import ClaudeClient
    from .config import Settings

logger = logging.getLogger("chatbot.presence")

UTC = timezone.utc  # datetime.UTC n'existe qu'à partir de Python 3.11
SOURCES_REFRESH_INTERVAL = 24 * 3600  # secondes
CUSTOM_STATUS_MAX_LENGTH = 128  # limite de Discord
RECENT_CUSTOM_STATUSES = 4  # statuts personnalisés récents, à ne pas répéter
GAME_ATTEMPTS = 5  # jeux essayés avant d'abandonner (fiche introuvable, jeu pour adultes…)

# Durée de chaque statut, tirée au sort entre ces bornes.
ANIME_EPISODES = (1, 5)
MUSIC_TRACKS = (6, 20)
DEFAULT_ANIME_EPISODES = 12  # épisodes disponibles quand AniList ne le précise pas
MINUTES = {  # en minutes
    "jeu": (45, 120),
    "en_ligne": (15, 45),
    "pensee": (30, 90),
    "absent": (10, 60),
    "ne_pas_deranger": (30, 90),
}

WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


class Kind(Enum):
    # Les valeurs sont les noms utilisés dans PRESENCE_WEIGHTS.
    ANIME = "anime"
    MUSIC = "musique"
    THOUGHT = "pensee"
    GAME = "jeu"
    ONLINE = "en_ligne"
    DND = "ne_pas_deranger"
    IDLE = "absent"
    NIGHT = "nuit"  # absent automatiquement pendant PRESENCE_NIGHT, jamais tiré au sort


@dataclass(frozen=True)
class Status:
    kind: Kind
    title: str | None = None    # anime, jeu, pensée, ou raison d'une absence
    track: Track | None = None  # musique en cours
    episode: int | None = None  # épisode en cours (None pour un film)
    details: str | None = None  # présentation du jeu, pour Claude uniquement

    def activity(self) -> discord.BaseActivity | None:
        if self.kind is Kind.ANIME:
            state = f"Épisode {self.episode}" if self.episode else None
            return discord.Activity(type=discord.ActivityType.watching, name=self.title, state=state)
        if self.kind is Kind.MUSIC:
            return discord.Activity(type=discord.ActivityType.listening,
                                    name=f"{self.track.title} de {self.track.artist}")
        if self.kind is Kind.GAME:
            return discord.Game(name=self.title)
        if self.title:  # pensée, ou raison d'une absence
            return discord.CustomActivity(name=self.title)
        return None

    def discord_status(self) -> discord.Status:
        if self.kind is Kind.DND:
            return discord.Status.dnd
        if self.kind in (Kind.IDLE, Kind.NIGHT):
            return discord.Status.idle
        return discord.Status.online

    def describe(self) -> str | None:
        """Ce que fait le bot, vu de lui-même (None : en ligne, sans activité)."""
        reason = f" (statut affiché : « {self.title} »)" if self.title else ""
        if self.kind is Kind.ANIME:
            episode = f"l'épisode {self.episode} de " if self.episode else ""
            return f"tu regardes {episode}l'anime « {self.title} »"
        if self.kind is Kind.MUSIC:
            return f"tu écoutes « {self.track.title} » de {self.track.artist}"
        if self.kind is Kind.GAME:
            details = f" (présentation du jeu sur Steam : {self.details})" if self.details else ""
            return f"tu joues à « {self.title} » sur Steam{details}"
        if self.kind is Kind.THOUGHT:
            return f"tu affiches le statut personnalisé « {self.title} »"
        if self.kind is Kind.DND:
            return f"tu es en « Ne pas déranger »{reason}"
        if self.kind is Kind.IDLE:
            return f"tu es absente{reason}"
        if self.kind is Kind.NIGHT:
            return "tu es absente pour la nuit"
        return None


# Un statut et sa durée en secondes. Un anime ou une musique en enchaîne plusieurs (un par épisode
# ou par morceau) : c'est le « programme » du statut.
Step = tuple[Status, float]


def format_duration(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d}" if minutes % 60 else f"{minutes // 60} h"


class PresenceManager:
    def __init__(self, bot: commands.Bot, claude: ClaudeClient, settings: Settings):
        self.bot = bot
        self.claude = claude
        self.settings = settings
        self.weights = {Kind(name): weight for name, weight in settings.presence_weights.items()}
        self.sources = PresenceSources(
            settings.history_file.with_name("presence_sources.json"),
            settings.deezer_playlists, settings.steam_api_key,
        )
        self.status = Status(Kind.ONLINE)
        self.program: list[Step] = []  # étapes restantes du statut en cours
        self.step_end = datetime.min.replace(tzinfo=UTC)  # fin de l'étape en cours (UTC)
        self.status_end: datetime | None = None  # fin du statut en cours, toutes étapes comprises
        self.recent_custom: deque[str] = deque(maxlen=RECENT_CUSTOM_STATUSES)
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if not self.sources.steam_enabled:
            logger.info("STEAM_API_KEY absent : pas de statut « joue à ».")
        self._tasks.append(asyncio.create_task(self._run()))

    def stop(self) -> None:
        for task in self._tasks:
            task.cancel()

    def prompt_note(self) -> str | None:
        """Ligne ajoutée au début du message envoyé à Claude."""
        description = self.status.describe()
        if not description:
            return None
        if self.status_end and self.status.kind is not Kind.NIGHT:
            now = datetime.now(UTC)
            speed = self.settings.presence_speed
            total = format_duration((self.status_end - now).total_seconds() * speed)
            if not self.program:
                if self.status.kind is Kind.ANIME and self.status.episode:
                    description += " (le dernier épisode que tu comptes regarder)"
                elif self.status.kind is Kind.MUSIC:
                    description += " (le dernier morceau que tu comptes écouter)"
                description += f", encore pendant environ {total}"
            else:
                # Anime ou musique : l'épisode (ou le morceau) en cours n'est qu'une partie du programme.
                current = format_duration((self.step_end - now).total_seconds() * speed)
                if self.status.kind is Kind.ANIME:
                    description += (f" (fin de l'épisode dans environ {current}) ; tu comptes regarder "
                                    f"jusqu'à l'épisode {self.program[-1][0].episode}, encore environ {total} en tout")
                else:
                    description += (f" (fin du morceau dans environ {current}) ; encore {len(self.program)} "
                                    f"morceau(x) après celui-ci, environ {total} de musique en tout")
        return f"[ton statut Discord : {description}]"

    async def apply(self) -> None:
        """Envoie le statut à Discord (aussi rappelé après une reconnexion, qui l'efface)."""
        try:
            await self.bot.change_presence(activity=self.status.activity(), status=self.status.discord_status())
        except Exception as exc:  # déconnecté : on_ready le renverra
            logger.warning("Statut Discord non appliqué : %s", exc)

    # ------------------------------------------------------------ boucles

    async def _run(self) -> None:
        self.sources.load_cache()
        await self.sources.refresh()
        self._tasks.append(asyncio.create_task(self._refresh_loop()))
        await self.bot.wait_until_ready()
        while True:
            try:
                wake = await self._step()
            except Exception:
                logger.exception("Erreur inattendue dans la gestion du statut")
                wake = datetime.now(UTC) + timedelta(minutes=1)
            await asyncio.sleep(max(1.0, (wake - datetime.now(UTC)).total_seconds()))

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(SOURCES_REFRESH_INTERVAL)
            try:
                await self.sources.refresh()
            except Exception:
                logger.exception("Erreur inattendue pendant la mise à jour des listes de statuts")

    async def _step(self) -> datetime:
        """Passe à l'étape suivante quand la précédente est finie ; renvoie l'heure du prochain réveil."""
        night = self.settings.presence_night
        now = datetime.now(UTC)
        local = now.astimezone(self.settings.timezone)

        if self.is_night(local):
            if self.status.kind is not Kind.NIGHT:
                # Coupure nette : le statut en cours s'arrête, un nouveau sera tiré au réveil.
                self.program = []
                self.status_end = None
                await self.set_status(Status(Kind.NIGHT))
            self.step_end = self.next_hour(local, night[1])
            return self.step_end

        if now >= self.step_end:
            if not self.program:
                self.program = await self.draw()
                total = sum(seconds for _, seconds in self.program)
                self.status_end = now + timedelta(seconds=total)
                logger.info("Nouveau statut pour %s.", format_duration(total * self.settings.presence_speed))
            status, seconds = self.program.pop(0)
            self.step_end = now + timedelta(seconds=seconds)
            await self.set_status(status)

        if night:
            return min(self.step_end, self.next_hour(local, night[0]))
        return self.step_end

    # ------------------------------------------------------------ tirage

    async def draw(self) -> list[Step]:
        """Tire un nouveau statut et son programme (durées divisées par PRESENCE_SPEED)."""
        available = {kind: weight for kind, weight in self.weights.items() if weight > 0 and self.has_source(kind)}
        # Jamais deux fois le même statut à la suite, sauf s'il n'y a pas d'autre choix.
        if len(available) > 1:
            available.pop(self.status.kind, None)
        kind = random.choices(list(available), weights=list(available.values()))[0] if available else Kind.ONLINE
        speed = self.settings.presence_speed

        if kind is Kind.ANIME:
            anime = self.pick(self.sources.anime, lambda a: a.title == self.status.title)
            if anime.episodes == 1:  # film
                return [(Status(kind, title=anime.title), anime.duration * 60 / speed)]
            available_episodes = anime.episodes or DEFAULT_ANIME_EPISODES
            count = min(random.randint(*ANIME_EPISODES), available_episodes)
            first = random.randint(1, available_episodes - count + 1)
            return [(Status(kind, title=anime.title, episode=n), anime.duration * 60 / speed)
                    for n in range(first, first + count)]

        if kind is Kind.MUSIC:
            count = min(random.randint(*MUSIC_TRACKS), len(self.sources.music))
            tracks = random.sample(self.sources.music, count)
            return [(Status(kind, track=t), t.duration / speed) for t in tracks]

        seconds = random.randint(*MINUTES[kind.value]) * 60 if kind.value in MINUTES else 0
        if kind is Kind.GAME:
            game = await self.pick_game()
            if game is None:
                return [(Status(Kind.ONLINE), seconds / speed)]
            return [(Status(kind, title=game.name, details=game.description), seconds / speed)]
        if kind in (Kind.THOUGHT, Kind.IDLE, Kind.DND):
            text = await self.write_custom_status(kind, seconds)
            if kind is Kind.THOUGHT and not text:
                kind = Kind.ONLINE  # pensée impossible à écrire : en ligne, sans activité
            return [(Status(kind, title=text), seconds / speed)]
        return [(Status(kind), seconds / speed)]

    async def pick_game(self) -> Game | None:
        """Jeu au hasard dans toute la boutique Steam, en écartant les jeux pour adultes."""
        for appid in random.sample(self.sources.games, min(GAME_ATTEMPTS, len(self.sources.games))):
            try:
                game = await self.sources.game_details(appid)
            except SourceError as exc:
                logger.warning("Fiche du jeu Steam %s illisible : %s", appid, exc)
                return None
            if game:
                return game
            logger.info("Jeu Steam %s écarté (introuvable, pas un jeu ou pour adultes).", appid)
        return None

    async def write_custom_status(self, kind: Kind, seconds: int) -> str | None:
        local = datetime.now(self.settings.timezone)
        now = f"{WEEKDAYS[local.weekday()]} {local.strftime('%d/%m/%Y, %H:%M')}"
        text = await self.claude.write_custom_status(
            kind.value, now, format_duration(seconds), list(self.recent_custom), CUSTOM_STATUS_MAX_LENGTH,
        )
        if text:
            self.recent_custom.append(text)
        return text

    def has_source(self, kind: Kind) -> bool:
        if kind is Kind.ANIME:
            return bool(self.sources.anime)
        if kind is Kind.MUSIC:
            return bool(self.sources.music)
        if kind is Kind.GAME:
            return bool(self.sources.games)
        return True

    @staticmethod
    def pick(items: list, is_current):
        """Élément au hasard, différent de l'actuel quand c'est possible."""
        return random.choice([item for item in items if not is_current(item)] or items)

    async def set_status(self, status: Status) -> None:
        self.status = status
        logger.info("Statut Discord : %s", status.describe() or "en ligne, sans activité")
        await self.apply()

    # ------------------------------------------------------------ nuit

    def is_night(self, local: datetime) -> bool:
        if not self.settings.presence_night:
            return False
        start, end = self.settings.presence_night
        if start < end:
            return start <= local.hour < end
        return local.hour >= start or local.hour < end  # nuit à cheval sur minuit (ex. 23-6)

    @staticmethod
    def next_hour(local: datetime, hour: int) -> datetime:
        """Prochain passage à hour:00, en UTC."""
        candidate = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= local:
            candidate += timedelta(days=1)
        return candidate.astimezone(UTC)
