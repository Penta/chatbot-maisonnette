"""Chargement et validation de la configuration depuis le fichier .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(Exception):
    """Erreur de configuration (variable manquante ou invalide)."""


def _get_str(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        if required:
            raise ConfigError(f"La variable {name} est obligatoire mais absente du fichier .env.")
        return default
    return value.strip()


def _get_int(name: str, default: int | None = None, required: bool = False, minimum: int | None = None) -> int | None:
    raw = _get_str(name, required=required)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"La variable {name} doit être un nombre entier (valeur actuelle : {raw!r}).")
    if minimum is not None and value < minimum:
        raise ConfigError(f"La variable {name} doit être supérieure ou égale à {minimum} (valeur actuelle : {value}).")
    return value


def _get_float(name: str, default: float | None = None) -> float | None:
    raw = _get_str(name)
    if raw is None:
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        raise ConfigError(f"La variable {name} doit être un nombre (valeur actuelle : {raw!r}).")


@dataclass(frozen=True)
class Settings:
    # Identifiants
    discord_token: str
    anthropic_api_key: str
    channel_id: int

    # Claude
    model: str
    max_tokens: int
    thinking: bool          # réflexion préalable avant de répondre (plus lent et plus cher)
    effort: str | None      # None = valeur par défaut du modèle
    api_timeout: float

    # Mémoire et contexte
    max_history_length: int
    context_message_limit: int
    history_file: Path
    personality_file: Path

    # Images
    max_image_size: int

    # Divers
    timezone: ZoneInfo
    price_input_per_mtok: float
    price_output_per_mtok: float
    log_file: Path


def load_settings() -> Settings:
    """Lit le fichier .env et renvoie une configuration validée."""
    # override=True : le fichier .env est prioritaire sur les variables d'environnement de Windows.
    load_dotenv(override=True)

    tz_name = _get_str("TIMEZONE", default="Europe/Paris")
    try:
        timezone = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise ConfigError(f"Fuseau horaire inconnu dans TIMEZONE : {tz_name!r} (exemple valide : Europe/Paris).")

    thinking_raw = (_get_str("THINKING", default="off") or "off").lower()
    if thinking_raw not in ("on", "off"):
        raise ConfigError(f"THINKING doit valoir « on » ou « off » (valeur actuelle : {thinking_raw!r}).")
    thinking = thinking_raw == "on"

    effort = _get_str("EFFORT")
    if effort is not None:
        effort = effort.lower()
        if effort not in EFFORT_LEVELS:
            raise ConfigError(f"EFFORT doit valoir {', '.join(EFFORT_LEVELS)} (valeur actuelle : {effort!r}).")
        if not thinking and effort in ("xhigh", "max"):
            raise ConfigError(f"EFFORT={effort} nécessite THINKING=on.")

    return Settings(
        discord_token=_get_str("DISCORD_TOKEN", required=True),
        anthropic_api_key=_get_str("ANTHROPIC_API_KEY", required=True),
        channel_id=_get_int("CHANNEL_ID", required=True),
        model=_get_str("CLAUDE_MODEL", default="claude-sonnet-5-5"),
        max_tokens=_get_int("MAX_TOKENS", default=1500, minimum=1),
        thinking=thinking,
        effort=effort,
        api_timeout=_get_float("API_TIMEOUT", default=60.0),
        max_history_length=_get_int("MAX_HISTORY_LENGTH", default=10, minimum=2),
        context_message_limit=_get_int("CONTEXT_MESSAGE_LIMIT", default=20, minimum=0),
        history_file=Path(_get_str("HISTORY_FILE", default="conversation_history.json")),
        personality_file=Path(_get_str("PERSONALITY_FILE", default="personality_prompt.txt")),
        # Taille maximale acceptée depuis Discord ; les grosses images sont réduites avant l'envoi à Claude.
        max_image_size=_get_int("MAX_IMAGE_SIZE", default=25_000_000, minimum=1),
        timezone=timezone,
        price_input_per_mtok=_get_float("PRICE_INPUT_PER_MTOK", default=2.0),
        price_output_per_mtok=_get_float("PRICE_OUTPUT_PER_MTOK", default=10.0),
        log_file=Path(_get_str("LOG_FILE", default="bot.log")),
    )
