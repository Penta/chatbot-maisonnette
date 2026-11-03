"""Point d'entrée : lance le bot avec `python main.py`."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import aiohttp
import discord

from bot.config import ConfigError, load_settings
from bot.discord_bot import ChatBot


def setup_logging(log_file: Path | None) -> None:
    """Journal sur la console et dans log_file (seulement sur la console si log_file vaut None)."""
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s : %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # discord.py et le SDK Anthropic sont bavards : on ne garde que leurs avertissements.
    # Le SDK 2.x passe par httpx2 ; httpx reste pour les versions plus anciennes.
    for noisy in ("discord", "httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        # Aussi dans le fichier de journal quand LOG_FILE est utilisable : un redémarrage raté y laisse une trace.
        setup_logging(exc.log_file)
        if len(exc.errors) == 1:
            logging.critical("Erreur de configuration : %s", exc)
        else:
            logging.critical("%d erreurs de configuration :\n%s", len(exc.errors),
                             "\n".join(f"  - {error}" for error in exc.errors))
        sys.exit(1)

    setup_logging(settings.log_file)
    # Rika n'utilise pas le vocal : inutile de signaler l'absence de PyNaCl et de davey.
    discord.VoiceClient.warn_nacl = False
    discord.VoiceClient.warn_dave = False
    bot = ChatBot(settings)
    try:
        bot.run(settings.discord_token, log_handler=None)
    except discord.LoginFailure:
        logging.critical("Token Discord invalide : vérifie DISCORD_TOKEN dans le .env.")
        sys.exit(1)
    except discord.PrivilegedIntentsRequired:
        logging.critical(
            "L'intent « Message Content » n'est pas activé. Active-le dans le portail développeur "
            "Discord (onglet Bot > Privileged Gateway Intents)."
        )
        sys.exit(1)
    except discord.DiscordServerError as exc:
        logging.critical("Discord semble en panne (erreur %s) : réessaie dans quelques minutes.", exc.status)
        sys.exit(1)
    except discord.HTTPException as exc:
        logging.critical("Discord a refusé une requête au démarrage (erreur %s : %s).",
                         exc.status, exc.text or "sans détail")
        sys.exit(1)
    # aiohttp.ClientError : pas de réseau, nom de domaine introuvable, connexion coupée ; TimeoutError : pas de réponse.
    except (aiohttp.ClientError, TimeoutError) as exc:
        logging.critical("Impossible de joindre Discord : vérifie la connexion Internet (%s).",
                         str(exc) or type(exc).__name__)
        sys.exit(1)
    # Clé API ou modèle refusés (ChatBot.setup_hook), salon dédié inutilisable (ChatBot.announce) :
    # déjà expliqué dans le journal.
    if bot.config_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
