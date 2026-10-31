"""Outils de mise en forme du texte pour Discord."""

from __future__ import annotations

import re

DISCORD_LIMIT = 2000

_FENCE_RE = re.compile(r"^\s*```(\S*)")

# Bloc dans lequel le modèle écrit parfois sa réflexion au lieu de la garder pour lui.
_REASONING_BLOCK = re.compile(r"\s*<(think|thinking|reasoning|function_calls|invoke)\b[^>]*>.*?</\1\s*>",
                              re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Retire la réflexion écrite au début de la réponse (<think>…</think>, faux appel d'outil…).

    Seul un bloc complet tout au début est retiré : c'est là que le modèle l'écrit, et une
    balise plus loin dans le texte (exemple, code) peut être voulue."""
    while match := _REASONING_BLOCK.match(text):
        text = text[match.end():]
    return text.strip()


def split_message(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Découpe un long texte en morceaux de `limit` caractères maximum.

    - coupe de préférence entre les lignes, puis entre les mots ;
    - si un bloc de code (```) est coupé, il est refermé à la fin du morceau
      et rouvert (avec le même langage) au début du suivant.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    open_fence: str | None = None  # ligne d'ouverture du bloc de code en cours, ex. "```python"

    def closing_len() -> int:
        return len("\n```") if open_fence is not None else 0

    def flush() -> None:
        nonlocal current
        body = current.rstrip("\n")
        if open_fence is not None:
            body += "\n```"
        if body.strip():
            chunks.append(body)
        current = (open_fence + "\n") if open_fence is not None else ""

    for line in text.split("\n"):
        # Une ligne trop longue à elle seule est découpée en morceaux plus petits.
        for piece in _split_long_line(line, limit - 10 - (len(open_fence) if open_fence else 0)):
            addition = piece + "\n"
            if len(current) + len(addition) + closing_len() > limit:
                flush()
            current += addition

        match = _FENCE_RE.match(line)
        if match:
            open_fence = None if open_fence is not None else line.strip()

    # Dernier morceau : pas de bloc à refermer artificiellement s'il est déjà fermé.
    body = current.rstrip("\n")
    if body.strip() and body.strip() != (open_fence or ""):
        chunks.append(body)
    return chunks


def _split_long_line(line: str, limit: int) -> list[str]:
    if len(line) <= limit:
        return [line]
    pieces: list[str] = []
    while len(line) > limit:
        cut = line.rfind(" ", 0, limit)
        if cut <= limit // 2:  # pas d'espace raisonnable : coupe nette
            cut = limit
        pieces.append(line[:cut].rstrip())
        line = line[cut:].lstrip()
    if line:
        pieces.append(line)
    return pieces
