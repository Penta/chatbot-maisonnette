"""Erreurs dont le message peut être affiché tel quel sur Discord."""

from __future__ import annotations


class UserFacingError(Exception):
    """Erreur avec un message affichable sur Discord (user_message)."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message
