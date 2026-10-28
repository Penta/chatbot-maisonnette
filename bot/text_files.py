"""Lecture des fichiers texte (.txt) joints aux messages."""

from __future__ import annotations

TEXT_EXTENSIONS = (".txt",)
# Taille maximale d'un fichier .txt (environ 25 000 tokens, pour limiter le coût d'un message).
TEXT_FILE_MAX_BYTES = 100_000


class TextFileError(Exception):
    """Fichier texte illisible ou refusé, avec un message affichable sur Discord."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def decode_text_file(data: bytes) -> str:
    """Décode le contenu d'un fichier .txt (UTF-8, sinon Windows-1252)."""
    if b"\x00" in data:
        raise TextFileError("Ce fichier ne contient pas du texte lisible.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Fichiers enregistrés par d'anciens éditeurs Windows (Bloc-notes, etc.).
        text = data.decode("cp1252", errors="replace")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise TextFileError("Ce fichier texte est vide.")
    return text
