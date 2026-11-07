"""Messages spontanés : le bot lance de lui-même la conversation dans son salon.

L'occasion vient de son statut : il vient de finir son anime ou sa partie de jeu (voir finished_activity),
et a donc quelque chose à raconter. Il ne parle que si le salon est calme depuis CALM, une fois sur
CHANCE environ, au plus SPONTANEOUS_PER_DAY fois par 24 h, et jamais deux fois de suite sans que
quelqu'un ait écrit entre-temps (voir SpontaneousLog.allowed). Claude peut aussi choisir de se taire
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
CALM = timedelta(hours=3)  # sans aucun message dans le salon (divisé par PRESENCE_SPEED)
CHANCE = 1 / 3  # chances de parler à chaque occasion : imprévisible, et pas à chaque fin d'épisode
DAY = timedelta(hours=24)
RECENT_AUTHORS = 30  # derniers messages du salon dont les auteurs voient leur fiche jointe
SILENCE = "RIEN"  # réponse de Claude quand rien de naturel ne lui vient
_SILENCE = re.compile(rf"\W*{SILENCE}\W*", re.IGNORECASE)
TOPICS = (Kind.ANIME, Kind.GAME)  # activités dont la fin donne envie d'en parler


def finished_activity(before: Status, after: Status) -> bool:
    """Le bot vient de finir son anime (dernier épisode prévu) ou sa partie de jeu, et reste disponible."""
    if before.kind not in TOPICS or not after.available:
        return False
    # Un anime enchaîne ses épisodes : l'épisode suivant du même anime n'est pas une fin.
    return (after.kind, after.title) != (before.kind, before.title)


def is_silence(text: str) -> bool:
    """Claude a choisi de ne rien dire (SILENCE, avec ou sans ponctuation, ou rien du tout)."""
    return not text.strip() or _SILENCE.fullmatch(text) is not None


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

    def allowed(self, now: datetime, last_message_id: int | None) -> bool:
        """Le bot peut-il parler de lui-même maintenant ? last_message_id : dernier message du salon, de
        n'importe qui (None : salon vide)."""
        if len([1 for _, date in self.sent if now - date < DAY]) >= self.per_day:
            return False
        if last_message_id is None:
            return True
        if self.sent and self.sent[-1][0] == last_message_id:
            return False  # personne n'a écrit depuis son dernier message spontané : il ne parle pas seul
        return now - discord.utils.snowflake_time(last_message_id) >= self.calm

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
