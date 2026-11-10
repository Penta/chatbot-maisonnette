"""Longs blocs de code des réponses : envoyés en fichier joint plutôt qu'en une suite de messages, et résumés
dans la mémoire de conversation pour ne pas être relus (et refacturés) à chaque message suivant."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import discord

from .text_utils import split_message

# Au-delà de l'une de ces tailles, un bloc de code part en fichier joint ; en dessous (un extrait de quelques
# lignes), il reste dans le message, lisible sans rien ouvrir.
MIN_LINES = 25
MIN_CHARS = 1500
# Fichiers joints au plus par message Discord : les blocs suivants restent dans le texte.
MAX_FILES = 10

# Marqueur ajouté par claude_client.py à une réponse coupée par MAX_TOKENS.
TRUNCATED = " […]"

# Pas de ``` plus loin sur la ligne : « ```let x = 1``` » s'ouvre et se referme sur elle-même, ce n'est pas un bloc.
_OPENING = re.compile(r"^\s*```((?:(?!```).)*)$")
_CLOSING = re.compile(r"^\s*```\s*$")
# Nom de fichier écrit sur la ligne d'ouverture (« ```html index.html ») : pas de chemin, une extension.
_FILE_NAME = re.compile(r"^[\w-][\w.-]{0,59}\.[A-Za-z0-9]{1,10}$")
_BLANK_LINES = re.compile(r"\n{3,}")

# Extension du fichier selon le langage du bloc, quand Claude n'a pas donné de nom.
_EXTENSIONS = {
    "html": "html", "htm": "html", "css": "css", "scss": "scss", "js": "js", "javascript": "js", "jsx": "jsx",
    "ts": "ts", "typescript": "ts", "tsx": "tsx", "json": "json", "py": "py", "python": "py", "sh": "sh",
    "bash": "sh", "shell": "sh", "ps1": "ps1", "powershell": "ps1", "bat": "bat", "sql": "sql", "java": "java",
    "kotlin": "kt", "kt": "kt", "c": "c", "h": "h", "cpp": "cpp", "c++": "cpp", "cs": "cs", "csharp": "cs",
    "php": "php", "rb": "rb", "ruby": "rb", "go": "go", "rust": "rs", "rs": "rs", "swift": "swift", "lua": "lua",
    "yaml": "yml", "yml": "yml", "toml": "toml", "ini": "ini", "xml": "xml", "svg": "svg", "md": "md",
    "markdown": "md", "dockerfile": "dockerfile", "vue": "vue", "svelte": "svelte",
}


@dataclass
class CodeFile:
    name: str
    content: str


@dataclass
class Prepared:
    """Réponse prête à envoyer : text (sans les longs blocs), files (ces blocs), et memory, le texte gardé dans
    la mémoire de conversation (chaque bloc y est remplacé par un marqueur qui nomme son fichier)."""
    text: str
    files: list[CodeFile] = field(default_factory=list)
    memory: str = ""


def prepare(answer: str) -> Prepared:
    """Retire de la réponse les blocs de code trop longs pour un message, qui partiront en fichier joint."""
    lines = answer.split("\n")
    shown: list[str] = []  # texte envoyé
    kept: list[str] = []  # texte gardé en mémoire
    files: list[CodeFile] = []
    truncated = False
    i = 0
    while i < len(lines):
        opening = _OPENING.match(lines[i])
        if not opening:
            shown.append(lines[i])
            kept.append(lines[i])
            i += 1
            continue
        end = next((k for k in range(i + 1, len(lines)) if _CLOSING.match(lines[k])), None)
        body = lines[i + 1:end] if end is not None else lines[i + 1:]
        content = "\n".join(body)
        # Bloc jamais refermé : la réponse a été coupée au milieu du code.
        cut = end is None and content.endswith(TRUNCATED)
        if cut:
            content = content[:-len(TRUNCATED)].rstrip()
        last = end if end is not None else len(lines) - 1
        if len(files) < MAX_FILES and (len(body) >= MIN_LINES or len(content) > MIN_CHARS):
            name = _unique(_file_name(opening.group(1)), {f.name for f in files})
            files.append(CodeFile(name, content + "\n"))
            details = f"{len(content.splitlines())} lignes" + (", coupé avant la fin" if cut else "")
            kept.append(f"[code envoyé en fichier joint : {name}, {details}]")
            truncated = truncated or cut
        else:
            shown.extend(lines[i:last + 1])
            kept.extend(lines[i:last + 1])
        i = last + 1
    if not files:
        return Prepared(answer, [], answer)
    text = _tidy(shown)
    if truncated:  # la coupure reste visible, comme pour une réponse sans fichier
        text = (text + TRUNCATED).strip()
    return Prepared(text, files, _tidy(kept))


def outgoing(text: str, files: list[CodeFile]) -> list[tuple[str | None, dict]]:
    """Messages Discord à envoyer, dans l'ordre : (texte, autres paramètres de channel.send). Les fichiers
    sont joints au dernier message, sous la fin de la réponse ; seuls s'il n'y a pas de texte."""
    messages: list[tuple[str | None, dict]] = [(chunk, {}) for chunk in split_message(text)]
    if files:
        # Un discord.File ne s'envoie qu'une fois : créés ici, au moment de l'envoi.
        attached = [discord.File(io.BytesIO(f.content.encode("utf-8")), filename=f.name) for f in files]
        if messages:
            messages[-1] = (messages[-1][0], {"files": attached})
        else:
            messages.append((None, {"files": attached}))
    return messages


def _file_name(info: str) -> str:
    """Nom du fichier d'après la ligne d'ouverture du bloc (« html index.html », « index.html », « html »)."""
    words = info.split()
    for word in words:
        if _FILE_NAME.match(word):
            return word
    language = words[0].lower() if words else ""
    return f"code.{_EXTENSIONS.get(language, 'txt')}"


def _unique(name: str, taken: set[str]) -> str:
    """Deux blocs du même nom (deux « code.js ») : le second devient « code-2.js »."""
    stem, dot, extension = name.rpartition(".")
    number = 2
    candidate = name
    while candidate in taken:
        candidate = f"{stem}-{number}.{extension}" if dot else f"{name}-{number}"
        number += 1
    return candidate


def _tidy(lines: list[str]) -> str:
    """Texte sans les lignes vides en trop laissées par les blocs retirés."""
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
