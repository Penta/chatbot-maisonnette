"""Lecture des fichiers texte (.txt, .md, .csv, code…) joints aux messages."""

from __future__ import annotations

import codecs

from .errors import UserFacingError

# Fichiers lus comme du texte brut : notes, données, sous-titres, configuration et code.
TEXT_EXTENSIONS = (
    ".txt", ".md", ".log", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".srt", ".vtt", ".html", ".htm", ".css", ".tex", ".sql",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".lua", ".sh", ".bat", ".ps1",
)
# Taille maximale d'un fichier .txt (environ 25 000 tokens, pour limiter le coût d'un message).
TEXT_FILE_MAX_BYTES = 100_000


class TextFileError(UserFacingError):
    """Fichier texte illisible ou refusé, avec un message affichable sur Discord."""


def _cp1252_fallback(error: UnicodeDecodeError) -> tuple[str, int]:
    """Relit en Windows-1252 les seuls octets qui ne sont pas de l'UTF-8 valide.

    Un octet isolé dans un fichier UTF-8 n'abîme ainsi que lui-même au lieu de
    déformer tous les accents du fichier.
    """
    return error.object[error.start:error.end].decode("cp1252", errors="replace"), error.end


codecs.register_error("rika_cp1252_fallback", _cp1252_fallback)


def _decode_invalid_utf8(data: bytes) -> str:
    """Décode un fichier qui n'est pas de l'UTF-8 valide : Windows-1252 en entier, ou mélangé.

    Dans un vrai fichier Windows-1252 (Bloc-notes « ANSI »), chaque accent est un octet
    invalide en UTF-8, mais deux ou trois octets voisins forment parfois un caractère
    UTF-8 valide par hasard (« é », espace insécable et « » » donnent « 頻 »). Le fichier
    n'est donc lu en mélangé (UTF-8, octets invalides en Windows-1252) que s'il a plus
    de caractères UTF-8 valides que d'octets invalides, ou s'il commence par un BOM UTF-8.
    """
    if not data.startswith(codecs.BOM_UTF8):
        lenient = data.decode("utf-8", errors="replace")  # un « � » par suite d'octets invalides
        # Sans les « � » déjà dans le fichier (EF BF BD est toujours décodé d'un bloc).
        invalid = lenient.count("�") - data.count("�".encode("utf-8"))
        valid = sum(1 for char in lenient if char > "\x7f") - invalid
        if invalid >= valid:
            return data.decode("cp1252", errors="replace")
    return data.decode("utf-8-sig", errors="rika_cp1252_fallback")


def decode_text_file(data: bytes) -> str:
    """Décode le contenu d'un fichier .txt (UTF-16 avec BOM, UTF-8, sinon Windows-1252 ou mélangé)."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        # « Unicode » du Bloc-notes : deux octets par caractère, dont beaucoup de \x00.
        try:
            text = data.decode("utf-16")
        except UnicodeDecodeError:
            raise TextFileError("Ce fichier ne contient pas du texte lisible.")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = _decode_invalid_utf8(data)
    if "\x00" in text:
        raise TextFileError("Ce fichier ne contient pas du texte lisible.")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise TextFileError("Ce fichier texte est vide.")
    return text
