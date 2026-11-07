"""Messages spontanés : le bot lance de lui-même la conversation dans son salon.

L'occasion vient de son statut : il vient de finir son anime ou sa partie de jeu (voir finished_activity),
et a donc quelque chose à raconter. Il ne parle que si aucun membre n'a écrit dans le salon depuis
CALM, une fois sur CHANCE environ, au plus SPONTANEOUS_PER_DAY fois par 24 h, et jamais deux fois de
suite sans qu'un membre ait écrit entre-temps (voir SpontaneousLog.allowed). Les messages des bots,
le sien compris (message « En ligne » au démarrage), ne comptent pas. Claude peut aussi choisir de se taire
(voir is_silence) : un message forcé sonne faux.

Les messages envoyés sont gardés dans spontaneous_messages.json : la limite et la règle « pas deux
fois de suite » tiennent après un redémarrage.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord

from .files import write_atomic
from .presence import Kind, Status

logger = logging.getLogger("chatbot.spontaneous")

UTC = timezone.utc
CALM = timedelta(hours=3)  # sans message d'un membre dans le salon (divisé par PRESENCE_SPEED)
CHANCE = 1 / 3  # chances de parler à chaque occasion : imprévisible, et pas à chaque fin d'épisode
DAY = timedelta(hours=24)
# Derniers messages du salon lus : on y cherche le dernier message d'un membre, et leurs auteurs voient leur
# fiche jointe. Au-delà, le salon est considéré comme calme.
RECENT_MESSAGES = 50
SILENCE = "RIEN"  # réponse de Claude quand rien de naturel ne lui vient
_SILENCE = re.compile(rf"\W*{SILENCE}\W*", re.IGNORECASE)  # le mot seul, en majuscules ou non
# Le mot en majuscules, accompagné d'une explication : seul sur une ligne, ou en tête suivi d'une ponctuation
# (« RIEN, je n'ai pas d'idée »). « RIEN ne vaut ce final » est un vrai message.
_SILENCE_LINE = re.compile(rf"^\W*{SILENCE}\W*$", re.MULTILINE)
_SILENCE_FIRST = re.compile(rf"\W*{SILENCE}\s*(?:[^\w\s]|$)")
TOPICS = (Kind.ANIME, Kind.GAME)  # activités dont la fin donne envie d'en parler


def finished_activity(before: Status, after: Status) -> bool:
    """Le bot vient de finir son anime (dernier épisode prévu) ou sa partie de jeu, et reste disponible."""
    if before.kind not in TOPICS or not after.available:
        return False
    # Un anime enchaîne ses épisodes : l'épisode suivant du même anime n'est pas une fin.
    return (after.kind, after.title) != (before.kind, before.title)


def is_silence(text: str) -> bool:
    """Claude a choisi de ne rien dire (SILENCE, avec ou sans ponctuation ni explication, ou rien du tout).
    Dans le doute, mieux vaut se taire : un silence ne se voit pas, « RIEN, je n'ai pas d'idée » dans le salon si."""
    return (not text.strip() or _SILENCE.fullmatch(text) is not None or _SILENCE_LINE.search(text) is not None
            or _SILENCE_FIRST.match(text) is not None)


def occasion_note(finished: Status, now: datetime) -> str:
    """Demande à Claude s'il veut parler de ce qu'il vient de finir (now : heure locale)."""
    if finished.kind is Kind.ANIME:
        episode = f"l'épisode {finished.episode} de " if finished.episode else ""
        what = f"tu viens de finir de regarder {episode}l'anime « {finished.title} »"
    else:
        details = f" (présentation du jeu sur Steam : {finished.details})" if finished.details else ""
        what = f"tu viens d'arrêter de jouer à « {finished.title} » sur Steam{details}"
    return (f"[Il est {now:%H:%M} : {what}, et personne n'a écrit dans le salon depuis un moment. Personne ne "
            "t'a parlé : si ça te donne envie d'en parler, écris de toi-même un court message, dans ton style, "
            "comme une pensée lancée en l'air (ce que tu en as pensé, une réaction, une question…). Tu peux "
            "t'adresser à un membre si tes notes disent que ça l'intéresse. Ne dis pas bonjour comme si tu "
            "ouvrais une conversation, ne fais pas remarquer que le salon est calme, et n'invente pas de détails "
            "que tu ne connais pas avec certitude. Si rien de naturel ne te vient, réponds seulement "
            f"{SILENCE}, sans rien d'autre : ce mot ne sera pas envoyé.]")


class SpontaneousLog:
    """Messages spontanés envoyés (identifiant du dernier morceau, date), gardés sur le disque."""

    def __init__(self, path: Path, per_day: int, speed: float = 1.0):
        self.path = path
        self.per_day = per_day
        self.calm = CALM / speed
        self.sent: list[tuple[int, datetime]] = self._load()

    def allowed(self, now: datetime, last_member_message: int | None) -> bool:
        """Le bot peut-il parler de lui-même maintenant ? last_member_message : identifiant du dernier message
        d'un membre dans le salon (None : aucun parmi les derniers messages)."""
        if len([1 for _, date in self.sent if now - date < DAY]) >= self.per_day:
            return False
        if self.sent and (last_member_message is None or last_member_message < self.sent[-1][0]):
            return False  # aucun membre n'a écrit depuis son dernier message spontané : il ne parle pas seul
        if last_member_message is None:
            return True
        return now - discord.utils.snowflake_time(last_member_message) >= self.calm

    def record(self, message_id: int, now: datetime) -> None:
        # Ceux des dernières 24 h comptent pour la limite ; le dernier, pour la règle « pas deux fois de suite ».
        self.sent = [entry for entry in self.sent if now - entry[1] < DAY] + [(message_id, now)]
        text = json.dumps([{"id": i, "date": date.isoformat()} for i, date in self.sent], indent=2)
        try:
            write_atomic(self.path, text)
        except OSError as exc:
            logger.warning("Impossible d'écrire %s : %s", self.path, exc)

    def _load(self) -> list[tuple[int, datetime]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            sent = [(int(entry["id"]), datetime.fromisoformat(entry["date"])) for entry in data]
            if any(date.tzinfo is None for _, date in sent):
                raise ValueError("date sans fuseau horaire")
            return sent
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("%s illisible, ignoré : %s", self.path, exc)
            return []
