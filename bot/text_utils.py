"""Outils de mise en forme du texte pour Discord."""

from __future__ import annotations

import re

DISCORD_LIMIT = 2000

# Pas de ``` plus loin sur la ligne : « ```let x = 1``` » s'ouvre et se referme sur elle-même, ce n'est pas un bloc.
_FENCE_RE = re.compile(r"^\s*```(?!.*```)(\S*)")
# Au-delà, le mot collé à ``` n'est pas un nom de langage : le bloc est rouvert avec ``` seul.
_MAX_LANGUAGE = 20

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
    open_fence: str | None = None  # marqueur du bloc de code en cours, ex. "```python" (sans le texte qui suit)

    def closing_len() -> int:
        return len("\n```") if open_fence is not None else 0

    def flush() -> None:
        nonlocal current
        body = current.rstrip("\n")
        if open_fence is not None:
            head, _, last = body.rpartition("\n")
            if last.strip() == open_fence:
                # Bloc ouvert en fin de morceau, encore vide : sa ligne d'ouverture passe au morceau suivant.
                body = head
            else:
                body += "\n```"
        if body.strip():
            chunks.append(body)
        current = (open_fence + "\n") if open_fence is not None else ""

    for line in text.split("\n"):
        match = _FENCE_RE.match(line)
        new_fence = _fence(match) if match and open_fence is None else None
        # Une ligne trop longue à elle seule est découpée en morceaux plus petits, assez courts pour
        # tenir avec la réouverture du bloc (en cours ou ouvert par cette ligne) et sa fermeture.
        fence = new_fence or open_fence
        pieces = _split_long_line(line, limit - 10 - (len(fence) if fence else 0))
        # Une ligne qui ouvre un bloc doit laisser la place de le refermer. Celle qui le ferme n'en a pas
        # besoin, sauf si elle est découpée : le bloc reste alors ouvert entre ses morceaux.
        if new_fence:
            room_to_close = len("\n```")
        elif match and len(pieces) == 1:
            room_to_close = 0
        else:
            room_to_close = closing_len()
        for piece in pieces:
            addition = piece + "\n"
            if len(current) + len(addition) + room_to_close > limit:
                flush()
            current += addition
            # Le bloc est ouvert dès le premier morceau de sa ligne d'ouverture : si elle est découpée,
            # la suite est dans le bloc, à refermer en fin de morceau.
            if new_fence:
                open_fence = new_fence

        if match and not new_fence:
            open_fence = None

    # Dernier morceau : pas de bloc à refermer artificiellement s'il est déjà fermé.
    body = current.rstrip("\n")
    if body.strip() and body.strip() != (open_fence or ""):
        chunks.append(body)
    return chunks


def _fence(match: re.Match[str]) -> str:
    """Marqueur d'ouverture d'un bloc (« ```py »), sans le texte qui suit sur la même ligne."""
    language = match.group(1)
    return "```" + (language if len(language) <= _MAX_LANGUAGE else "")


def _split_long_line(line: str, limit: int) -> list[str]:
    # Au moins un caractère par morceau, même si la place manque : sinon la ligne ne raccourcit jamais.
    limit = max(limit, 1)
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
