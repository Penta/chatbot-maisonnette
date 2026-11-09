"""Erreurs dont le message peut être affiché tel quel sur Discord."""

from __future__ import annotations

import aiohttp
import discord

# Discord n'a pas pu être joint : son erreur, ou une coupure réseau (OSError comprend les délais dépassés).
DISCORD_FAILURES = (discord.HTTPException, aiohttp.ClientError, OSError)


class UserFacingError(Exception):
    """Erreur avec un message affichable sur Discord (user_message)."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message
