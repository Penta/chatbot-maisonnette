"""Chargement et validation de la configuration depuis le fichier .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from .models import PRICES, thinking_off

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Chances (relatives) de chaque statut Discord lors d'un changement de statut.
DEFAULT_PRESENCE_WEIGHTS = {
    "anime": 20, "musique": 20, "pensee": 10, "jeu": 20, "en_ligne": 20, "ne_pas_deranger": 5, "absent": 5,
}
# Tops Deezer officiels : Top Worldwide, Top France, Top Japan.
DEFAULT_DEEZER_PLAYLISTS = "3155776842,1109890291,1362508955"


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


def _get_presence_weights() -> dict[str, int]:
    """PRESENCE_WEIGHTS, par exemple « anime=20,musique=20,… » ; les statuts absents gardent leur valeur par défaut."""
    weights = dict(DEFAULT_PRESENCE_WEIGHTS)
    raw = _get_str("PRESENCE_WEIGHTS")
    if raw is None:
        return weights
    for item in raw.split(","):
        name, _, value = item.partition("=")
        name = name.strip().lower()
        if name not in weights:
            raise ConfigError(f"Statut inconnu dans PRESENCE_WEIGHTS : {name!r} (statuts possibles : {', '.join(weights)}).")
        try:
            weights[name] = int(value)
        except ValueError:
            raise ConfigError(f"PRESENCE_WEIGHTS : la valeur de {name} doit être un nombre entier (valeur actuelle : {value.strip()!r}).")
        if weights[name] < 0:
            raise ConfigError(f"PRESENCE_WEIGHTS : la valeur de {name} ne peut pas être négative.")
    if sum(weights.values()) == 0:
        raise ConfigError("PRESENCE_WEIGHTS : au moins un statut doit avoir une valeur supérieure à 0.")
    return weights


def _get_night() -> tuple[int, int] | None:
    """PRESENCE_NIGHT, par exemple « 2-6 » (de 2 h à 6 h) ; « off » = pas de nuit."""
    raw = _get_str("PRESENCE_NIGHT", default="2-6")
    if raw.lower() == "off":
        return None
    start, _, end = raw.partition("-")
    try:
        hours = (int(start), int(end))
    except ValueError:
        hours = None
    if hours is None or not all(0 <= h <= 23 for h in hours) or hours[0] == hours[1]:
        raise ConfigError(f"PRESENCE_NIGHT doit être de la forme « 2-6 » (heures de 0 à 23) ou « off » (valeur actuelle : {raw!r}).")
    return hours


def _get_deezer_playlists() -> tuple[int, ...]:
    raw = _get_str("DEEZER_PLAYLISTS", default=DEFAULT_DEEZER_PLAYLISTS)
    try:
        return tuple(int(p) for p in raw.split(",") if p.strip())
    except ValueError:
        raise ConfigError(f"DEEZER_PLAYLISTS doit être une liste de numéros séparés par des virgules (valeur actuelle : {raw!r}).")


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

    # Accès à Internet (outils web_search et web_fetch exécutés par Anthropic)
    web_search: bool
    web_max_uses: int       # nombre maximal de recherches (et de pages lues) par message

    # Mémoire et contexte
    max_history_length: int
    context_message_limit: int
    history_file: Path
    personality_file: Path

    # Images
    max_image_size: int

    # Statut Discord (voir presence.py)
    presence: bool
    presence_speed: float                 # pour tester : 10 = statuts 10 fois plus courts
    presence_weights: dict[str, int]
    presence_night: tuple[int, int] | None  # heures de début et de fin ; None = pas de nuit
    deezer_playlists: tuple[int, ...]
    steam_api_key: str | None
    steam_id: str | None

    # Divers
    timezone: ZoneInfo
    price_input_per_mtok: float | None   # None = modèle absent de la table des tarifs
    price_output_per_mtok: float | None
    price_web_search_per_1000: float
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

    model = _get_str("CLAUDE_MODEL", default="claude-sonnet-5-5")
    effort = _get_str("EFFORT")
    if effort is not None:
        effort = effort.lower()
        if effort not in EFFORT_LEVELS:
            raise ConfigError(f"EFFORT doit valoir {', '.join(EFFORT_LEVELS)} (valeur actuelle : {effort!r}).")
        # Couper la réflexion n'est accepté qu'avec un effort high ou moins.
        if not thinking and effort in ("xhigh", "max") and thinking_off(model) is not None:
            raise ConfigError(f"EFFORT={effort} nécessite THINKING=on avec le modèle {model}.")

    # Tarifs du .env s'ils sont renseignés, sinon ceux du modèle choisi.
    default_input, default_output = PRICES.get(model, (None, None))

    web_search_raw = (_get_str("WEB_SEARCH", default="off") or "off").lower()
    if web_search_raw not in ("on", "off"):
        raise ConfigError(f"WEB_SEARCH doit valoir « on » ou « off » (valeur actuelle : {web_search_raw!r}).")

    presence_raw = (_get_str("PRESENCE", default="on") or "on").lower()
    if presence_raw not in ("on", "off"):
        raise ConfigError(f"PRESENCE doit valoir « on » ou « off » (valeur actuelle : {presence_raw!r}).")
    presence_speed = _get_float("PRESENCE_SPEED", default=1.0)
    if presence_speed <= 0:
        raise ConfigError(f"PRESENCE_SPEED doit être supérieur à 0 (valeur actuelle : {presence_speed}).")

    return Settings(
        discord_token=_get_str("DISCORD_TOKEN", required=True),
        anthropic_api_key=_get_str("ANTHROPIC_API_KEY", required=True),
        channel_id=_get_int("CHANNEL_ID", required=True),
        model=model,
        max_tokens=_get_int("MAX_TOKENS", default=1500, minimum=1),
        thinking=thinking,
        effort=effort,
        api_timeout=_get_float("API_TIMEOUT", default=60.0),
        web_search=web_search_raw == "on",
        web_max_uses=_get_int("WEB_MAX_USES", default=3, minimum=1),
        max_history_length=_get_int("MAX_HISTORY_LENGTH", default=10, minimum=2),
        context_message_limit=_get_int("CONTEXT_MESSAGE_LIMIT", default=20, minimum=0),
        history_file=Path(_get_str("HISTORY_FILE", default="conversation_history.json")),
        personality_file=Path(_get_str("PERSONALITY_FILE", default="personality_prompt.txt")),
        # Taille maximale acceptée depuis Discord ; les grosses images sont réduites avant l'envoi à Claude.
        max_image_size=_get_int("MAX_IMAGE_SIZE", default=25_000_000, minimum=1),
        presence=presence_raw == "on",
        presence_speed=presence_speed,
        presence_weights=_get_presence_weights(),
        presence_night=_get_night(),
        deezer_playlists=_get_deezer_playlists(),
        steam_api_key=_get_str("STEAM_API_KEY"),
        steam_id=_get_str("STEAM_ID"),
        timezone=timezone,
        price_input_per_mtok=_get_float("PRICE_INPUT_PER_MTOK", default=default_input),
        price_output_per_mtok=_get_float("PRICE_OUTPUT_PER_MTOK", default=default_output),
        price_web_search_per_1000=_get_float("PRICE_WEB_SEARCH_PER_1000", default=10.0),
        log_file=Path(_get_str("LOG_FILE", default="bot.log")),
    )
