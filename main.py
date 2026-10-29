"""Point d'entrée : lance le bot avec `python main.py`."""

import logging
import sys
from logging.handlers import RotatingFileHandler

import discord

from bot.config import ConfigError, load_settings
from bot.discord_bot import ChatBot


def setup_logging(log_file) -> None:
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s : %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # discord.py et le SDK Anthropic sont bavards : on ne garde que leurs avertissements.
    # Le SDK 2.x passe par httpx2 ; httpx reste pour les versions plus anciennes.
    for noisy in ("discord", "httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
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


if __name__ == "__main__":
    main()
