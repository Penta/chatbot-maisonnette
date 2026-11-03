"""Chargement et validation de la configuration depuis le fichier .env."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from .models import PRICES, thinking_off

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Tokens de réflexion à prévoir avant la réécriture d'un fichier de mémoire, selon EFFORT
# (None : valeur par défaut du modèle, soit high).
REFLECTION_TOKENS = {"low": 2000, "medium": 2000, "high": 4000, None: 4000, "xhigh": 8000, "max": 8000}

# Chances (relatives) de chaque statut Discord lors d'un changement de statut.
DEFAULT_PRESENCE_WEIGHTS = {
    "anime": 20, "musique": 20, "pensee": 10, "jeu": 20, "en_ligne": 20, "ne_pas_deranger": 5, "absent": 5,
}
# Tops Deezer officiels : Top Worldwide, Top France, Top Japan.
DEFAULT_DEEZER_PLAYLISTS = "3155776842,1109890291,1362508955"


class ConfigError(Exception):
    """Erreur de configuration (variable manquante ou invalide). Celle de load_settings rassemble toutes les
    erreurs trouvées (errors) et indique le fichier de journal s'il est utilisable (log_file, sinon None)."""

    def __init__(self, *errors: str, log_file: Path | None = None, log_level: str = "INFO"):
        super().__init__(*errors)
        self.errors = list(errors)
        self.log_file = log_file
        self.log_level = log_level

    def __str__(self) -> str:
        return "\n".join(self.errors)


def _get_str(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        if required:
            raise ConfigError(f"La variable {name} est obligatoire mais absente du fichier .env.")
        return default
    value = value.strip()
    # docker run --env-file garde les guillemets que python-dotenv retire : DISCORD_TOKEN="abc".
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value or default


def _get_int(name: str, default: int | None = None, required: bool = False, minimum: int | None = None,
             maximum: int | None = None) -> int | None:
    raw = _get_str(name, required=required)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"La variable {name} doit être un nombre entier (valeur actuelle : {raw!r}).")
    if minimum is not None and value < minimum:
        raise ConfigError(f"La variable {name} doit être supérieure ou égale à {minimum} (valeur actuelle : {value}).")
    if maximum is not None and value > maximum:
        raise ConfigError(f"La variable {name} ne peut pas dépasser {maximum} (valeur actuelle : {value}).")
    return value


def _get_float(name: str, default: float | None = None, minimum: float | None = None,
               maximum: float | None = None) -> float | None:
    raw = _get_str(name)
    if raw is None:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        raise ConfigError(f"La variable {name} doit être un nombre (valeur actuelle : {raw!r}).")
    if not math.isfinite(value):  # float() accepte « nan » et « inf »
        raise ConfigError(f"La variable {name} doit être un nombre (valeur actuelle : {raw!r}).")
    if minimum is not None and value < minimum:
        raise ConfigError(f"La variable {name} doit être supérieure ou égale à {minimum:g} (valeur actuelle : {value:g}).")
    if maximum is not None and value > maximum:
        raise ConfigError(f"La variable {name} ne peut pas dépasser {maximum:g} (valeur actuelle : {value:g}).")
    return value


def _check_output(name: str, path: Path, folder_only: bool = False) -> None:
    """Vérifie que le bot pourra écrire path (ou dans le dossier path si folder_only). Le dossier n'est pas
    créé : avec Docker, un dossier absent signale en général un volume mal monté, et les fichiers écrits
    dans le conteneur seraient perdus au redémarrage."""
    if not folder_only and path.is_dir():
        raise ConfigError(f"{name} désigne un dossier, pas un fichier : {path}.")
    folder = path if folder_only else path.parent
    if not folder.is_dir():
        raise ConfigError(f"Le dossier de {name} n'existe pas : {folder.resolve()} (volume non monté ?).")
    try:
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as exc:
        raise ConfigError(f"Impossible d'écrire dans le dossier de {name} ({folder.resolve()}) : {exc.strerror or exc}.")


def _get_presence_weights() -> dict[str, int]:
    """PRESENCE_WEIGHTS, par exemple « anime=20,musique=20,… » ; les statuts absents gardent leur valeur par défaut."""
    weights = dict(DEFAULT_PRESENCE_WEIGHTS)
    raw = _get_str("PRESENCE_WEIGHTS")
    if raw is None:
        return weights
    seen = set()
    for item in raw.split(","):
        if not item.strip():
            continue  # virgule finale ou doublée
        name, _, value = item.partition("=")
        name = name.strip().lower()
        if name not in weights:
            raise ConfigError(f"Statut inconnu dans PRESENCE_WEIGHTS : {name!r} (statuts possibles : {', '.join(weights)}).")
        if name in seen:
            raise ConfigError(f"PRESENCE_WEIGHTS : le statut {name} apparaît plusieurs fois.")
        seen.add(name)
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
    # Identifiants (repr=False : jamais affichés si la configuration est journalisée)
    discord_token: str = field(repr=False)
    anthropic_api_key: str = field(repr=False)
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

    # Mémoire des membres : fiches tenues par Claude (voir members.py)
    memory: bool
    memory_dir: Path
    memory_member_limit: int   # caractères maximum d'une fiche de membre
    memory_server_limit: int   # caractères maximum du fichier général
    memory_max_operations: int  # lectures et écritures maximum par réponse

    # Images et PDF
    max_image_size: int
    pdf_max_pages: int  # 0 : PDF refusés

    # Statut Discord (voir presence.py)
    presence: bool
    presence_speed: float                 # pour tester : 10 = statuts 10 fois plus courts
    presence_weights: dict[str, int]
    presence_night: tuple[int, int] | None  # heures de début et de fin ; None = pas de nuit
    deezer_playlists: tuple[int, ...]
    steam_api_key: str | None = field(repr=False)

    # Divers
    timezone: ZoneInfo
    price_input_per_mtok: float | None   # None = modèle absent de la table des tarifs
    price_output_per_mtok: float | None
    price_web_search_per_1000: float
    log_file: Path | None  # None = LOG_FILE=off, journal sur la console seulement
    log_level: str


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def load_settings() -> Settings:
    """Lit le fichier .env et renvoie une configuration validée. Toutes les erreurs sont rassemblées dans une
    seule ConfigError : avec trois variables fausses, un seul redémarrage suffit pour les voir."""
    # override=True : le fichier .env est prioritaire sur les variables d'environnement de Windows.
    load_dotenv(override=True)
    errors: list[str] = []

    def check(read, *args, **kwargs):
        """Appelle read ; en cas d'erreur, la note et renvoie None (les vérifications croisées l'ignorent)."""
        try:
            return read(*args, **kwargs)
        except ConfigError as exc:
            errors.extend(exc.errors)
            return None

    tz_name = _get_str("TIMEZONE", default="Europe/Paris")
    try:
        timezone = ZoneInfo(tz_name)
    # ValueError : nom mal formé (« Europe/Paris/ », « ../Paris ») ; OSError : nom trop long pour un fichier.
    except (ZoneInfoNotFoundError, ValueError, OSError):
        timezone = None
        errors.append(f"Fuseau horaire inconnu dans TIMEZONE : {tz_name!r} (exemple valide : Europe/Paris).")

    thinking_raw = (_get_str("THINKING", default="off") or "off").lower()
    if thinking_raw not in ("on", "off"):
        errors.append(f"THINKING doit valoir « on » ou « off » (valeur actuelle : {thinking_raw!r}).")
    thinking = thinking_raw == "on"

    model = _get_str("CLAUDE_MODEL", default="claude-sonnet-5-5")
    effort = _get_str("EFFORT")
    if effort is not None:
        effort = effort.lower()
        if effort not in EFFORT_LEVELS:
            errors.append(f"EFFORT doit valoir {', '.join(EFFORT_LEVELS)} (valeur actuelle : {effort!r}).")
        # Couper la réflexion n'est accepté qu'avec un effort high ou moins.
        elif thinking_raw == "off" and effort in ("xhigh", "max") and thinking_off(model) is not None:
            errors.append(f"EFFORT={effort} nécessite THINKING=on avec le modèle {model}.")

    # Tarifs du .env s'ils sont renseignés, sinon ceux du modèle choisi.
    default_input, default_output = PRICES.get(model, (None, None))

    web_search_raw = (_get_str("WEB_SEARCH", default="off") or "off").lower()
    if web_search_raw not in ("on", "off"):
        errors.append(f"WEB_SEARCH doit valoir « on » ou « off » (valeur actuelle : {web_search_raw!r}).")

    memory_raw = (_get_str("MEMORY", default="off") or "off").lower()
    if memory_raw not in ("on", "off"):
        errors.append(f"MEMORY doit valoir « on » ou « off » (valeur actuelle : {memory_raw!r}).")
    history_file = Path(_get_str("HISTORY_FILE", default="conversation_history.json"))
    # Vide : à côté de l'historique, comme les autres fichiers du bot (donc sur le même volume avec Docker).
    memory_dir = _get_str("MEMORY_DIR")
    # La fenêtre « Modifier » de /memoire contient au plus 5 zones de 4000 caractères.
    memory_limits = {name: check(_get_int, name, default=default, minimum=200, maximum=20000)
                     for name, default in (("MEMORY_MEMBER_LIMIT", 3000), ("MEMORY_SERVER_LIMIT", 5000))}

    presence_raw = (_get_str("PRESENCE", default="on") or "on").lower()
    if presence_raw not in ("on", "off"):
        errors.append(f"PRESENCE doit valoir « on » ou « off » (valeur actuelle : {presence_raw!r}).")
    # Au-delà, le statut change trop souvent (écriture de presence_state.json, limites de débit de Discord).
    presence_speed = check(_get_float, "PRESENCE_SPEED", default=1.0, maximum=100)
    if presence_speed is not None and presence_speed <= 0:
        errors.append(f"PRESENCE_SPEED doit être supérieur à 0 (valeur actuelle : {presence_speed}).")

    log_level = (_get_str("LOG_LEVEL", default="INFO") or "INFO").upper()
    if log_level not in LOG_LEVELS:
        errors.append(f"LOG_LEVEL doit valoir {', '.join(LOG_LEVELS)} (valeur actuelle : {log_level!r}).")
        log_level = "INFO"
    # off : en conteneur, la console suffit (OpenShift et Docker la collectent déjà).
    log_file_raw = _get_str("LOG_FILE", default="bot.log")

    # Une valeur invalide vaut None ici : l'erreur est notée, et la configuration n'est pas renvoyée.
    settings = Settings(
        discord_token=check(_get_str, "DISCORD_TOKEN", required=True),
        anthropic_api_key=check(_get_str, "ANTHROPIC_API_KEY", required=True),
        channel_id=check(_get_int, "CHANNEL_ID", required=True),
        model=model,
        # Une limite haute ne coûte que si elle est atteinte ; trop basse, elle coupe la réécriture
        # d'un fichier de mémoire plein (voir memory_max_tokens).
        # Au-delà de 64000, l'API refuse la requête (limite de sortie des modèles).
        max_tokens=check(_get_int, "MAX_TOKENS", default=4000, minimum=1, maximum=64000),
        thinking=thinking,
        effort=effort,
        api_timeout=check(_get_float, "API_TIMEOUT", default=60.0, minimum=1),
        web_search=web_search_raw == "on",
        # Chaque recherche est facturée : au-delà de 10, une seule question peut coûter cher.
        web_max_uses=check(_get_int, "WEB_MAX_USES", default=3, minimum=1, maximum=10),
        max_history_length=check(_get_int, "MAX_HISTORY_LENGTH", default=10, minimum=2),
        # Lus à chaque mention puis envoyés à Claude : au-delà, chaque mention coûte cher (ou dépasse la limite).
        context_message_limit=check(_get_int, "CONTEXT_MESSAGE_LIMIT", default=20, minimum=0, maximum=100),
        history_file=history_file,
        personality_file=Path(_get_str("PERSONALITY_FILE", default="personality_prompt.txt")),
        memory=memory_raw == "on",
        memory_dir=Path(memory_dir) if memory_dir else history_file.with_name("memories"),
        # Chaque fiche jointe est relue à chaque message : plus elle est longue, plus il coûte.
        memory_member_limit=memory_limits["MEMORY_MEMBER_LIMIT"],
        memory_server_limit=memory_limits["MEMORY_SERVER_LIMIT"],
        memory_max_operations=check(_get_int, "MEMORY_MAX_OPERATIONS", default=5, minimum=1, maximum=20),
        # Taille maximale acceptée depuis Discord ; les grosses images sont réduites avant l'envoi à Claude.
        max_image_size=check(_get_int, "MAX_IMAGE_SIZE", default=25_000_000, minimum=1),
        # Chaque page coûte son texte et son image (souvent plusieurs milliers de tokens).
        # L'API refuse les PDF de plus de 100 pages.
        pdf_max_pages=check(_get_int, "PDF_MAX_PAGES", default=20, minimum=0, maximum=100),
        presence=presence_raw == "on",
        presence_speed=presence_speed,
        presence_weights=check(_get_presence_weights),
        presence_night=check(_get_night),
        deezer_playlists=check(_get_deezer_playlists),
        steam_api_key=_get_str("STEAM_API_KEY"),
        timezone=timezone,
        price_input_per_mtok=check(_get_float, "PRICE_INPUT_PER_MTOK", default=default_input, minimum=0),
        price_output_per_mtok=check(_get_float, "PRICE_OUTPUT_PER_MTOK", default=default_output, minimum=0),
        price_web_search_per_1000=check(_get_float, "PRICE_WEB_SEARCH_PER_1000", default=10.0, minimum=0),
        log_file=None if log_file_raw.lower() == "off" else Path(log_file_raw),
        log_level=log_level,
    )
    count = len(errors)
    if settings.log_file is not None:
        check(_check_output, "LOG_FILE", settings.log_file)
    # Journal utilisable : main.py y écrit les erreurs de configuration, pas seulement sur la console.
    log_file = settings.log_file if len(errors) == count else None
    check(_check_output, "HISTORY_FILE", settings.history_file)
    if settings.memory:
        # Le dossier de la mémoire se crée au premier enregistrement : son parent doit déjà exister.
        if settings.memory_dir.is_dir():
            check(_check_output, "MEMORY_DIR", settings.memory_dir, folder_only=True)
        else:
            check(_check_output, "MEMORY_DIR", settings.memory_dir)
    # Sans clé Steam, « jeu » est retiré au tirage : s'il était le seul statut possible, le bot resterait
    # « en ligne » sans activité.
    if settings.presence and settings.presence_weights and not settings.steam_api_key and not any(
            weight for name, weight in settings.presence_weights.items() if name != "jeu"):
        errors.append("PRESENCE_WEIGHTS : le statut jeu est le seul possible, mais il nécessite STEAM_API_KEY. "
                      "Renseigne STEAM_API_KEY ou donne une valeur supérieure à 0 à un autre statut.")
    if (settings.memory and None not in (settings.max_tokens, *memory_limits.values())
            and settings.max_tokens < (minimum := memory_max_tokens(settings)[0])):
        errors.append(f"MAX_TOKENS ({settings.max_tokens}) est trop petit pour la mémoire des membres : la "
                      f"réécriture d'un fichier plein serait coupée et la note perdue. Mets au moins {minimum}, "
                      f"ou baisse MEMORY_SERVER_LIMIT et MEMORY_MEMBER_LIMIT.")
    if errors:
        raise ConfigError(*errors, log_file=log_file, log_level=log_level)
    return settings


def memory_max_tokens(settings: Settings) -> tuple[int, int]:
    """MAX_TOKENS nécessaire pour réécrire un fichier de mémoire plein d'un seul appel d'outil :
    (minimum sans réflexion, valeur conseillée avec la réflexion attendue)."""
    # Environ 2,5 caractères par token en français (estimation prudente), plus 2000 pour le reste de l'appel.
    minimum = max(settings.memory_member_limit, settings.memory_server_limit) * 2 // 5 + 2000
    # Sans THINKING=on, le modèle réfléchit quand même s'il ne permet pas de couper la réflexion.
    if not settings.thinking and thinking_off(settings.model) is not None:
        return minimum, minimum
    return minimum, minimum + REFLECTION_TOKENS[settings.effort]


def memory_warning(settings: Settings) -> str | None:
    """Avertissement à afficher au démarrage quand MAX_TOKENS laisse trop peu de place à la réflexion
    pour réécrire un fichier de mémoire plein (None : rien à signaler)."""
    if not settings.memory or settings.max_tokens >= (advised := memory_max_tokens(settings)[1]):
        return None
    return (f"MAX_TOKENS ({settings.max_tokens}) laisse peu de place à la réflexion (effort "
            f"{settings.effort or 'par défaut'}) : la réécriture d'un fichier de mémoire plein pourrait être "
            f"coupée et la note perdue. Mets au moins {advised} dans le .env.")
