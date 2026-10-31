"""Fichiers de mémoire tenus par Claude : une fiche par membre et un fichier général du serveur.

Outil maison plutôt que l'outil mémoire d'Anthropic (memory_20250818) : celui-ci ajoute au prompt
l'ordre de consulter tout le dossier avant chaque réponse, soit un appel de plus à chaque message,
alors que le bot joint déjà lui-même les fiches utiles."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .members import MemberRegistry, display_name

logger = logging.getLogger("chatbot.memory")

SERVER_FILE = "serveur.md"
MEMBER_FILE = re.compile(r"^membres/(\d+)\.md$")

MEMORY_TOOL = {
    "name": "memoire",
    "description": (
        "Lit ou modifie tes fichiers de mémoire : « serveur.md » (fichier général du serveur) et "
        "« membres/<identifiant>.md » (fiche d'un membre, chemin indiqué dans la liste des membres). "
        "Commandes : « lire » renvoie le fichier ; « ajouter » ajoute « texte » (une ou plusieurs lignes) "
        "à la fin du fichier, et le crée s'il n'existe pas ; « remplacer » remplace « ancien », recopié "
        "exactement et présent une seule fois dans le fichier, par « texte » (vide pour supprimer) ; "
        "« reecrire » remplace tout le fichier par « texte ». Les champs inutiles à la commande restent vides."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "commande": {"type": "string", "enum": ["lire", "ajouter", "remplacer", "reecrire"]},
            "fichier": {"type": "string", "description": "serveur.md ou membres/<identifiant>.md"},
            "texte": {"type": "string"},
            "ancien": {"type": "string"},
        },
        "required": ["commande", "fichier", "texte", "ancien"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    written: bool = False  # un fichier a été modifié


class MemoryStore:
    """Lit et écrit les fichiers de mémoire, dans les seuls chemins autorisés et sous leur taille maximale."""

    def __init__(self, root: Path, members: MemberRegistry, member_limit: int, server_limit: int):
        self.root = root
        self.members = members
        self.member_limit = member_limit
        self.server_limit = server_limit
        self._lock = asyncio.Lock()  # deux réponses en parallèle ne modifient pas un fichier en même temps

    # ------------------------------------------------------------------ accès

    def read(self, name: str) -> str:
        """Contenu d'un fichier de mémoire (vide s'il n'existe pas)."""
        try:
            return (self.root / name).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def member_file(self, member_id: int | str) -> str:
        return f"membres/{member_id}.md"

    async def run(self, command: dict) -> ToolResult:
        """Exécute une commande de l'outil « memoire »."""
        action = command.get("commande")
        name = self._normalize(command.get("fichier", ""))
        text = (command.get("texte") or "").strip()
        old = (command.get("ancien") or "").strip()

        limit, label = self._check(name)
        if limit is None:
            logger.warning("Mémoire : %s refusé sur %r (%s)", action, command.get("fichier"), label)
            return ToolResult(label, is_error=True)

        if action == "lire":
            content = self.read(name)
            return ToolResult(content or f"Le fichier {name} est vide : rien n'y est encore noté.")

        async with self._lock:
            before = self.read(name)
            if action == "ajouter":
                if not text:
                    return ToolResult("Rien à ajouter : « texte » est vide.", is_error=True)
                # Une ligne déjà présente telle quelle n'est pas ajoutée une deuxième fois.
                existing = {line.strip() for line in before.splitlines()}
                new_lines = [line for line in text.splitlines() if line.strip() not in existing]
                after = "\n".join([before, *new_lines]) if before else text
            elif action == "remplacer":
                if not old:
                    return ToolResult("« ancien » est vide : recopie exactement le passage à remplacer.", is_error=True)
                count = before.count(old)
                if count != 1:
                    problem = "est introuvable" if count == 0 else f"apparaît {count} fois"
                    return ToolResult(f"Rien n'a été remplacé : « {old} » {problem} dans {name}. "
                                      f"Recopie exactement un passage présent une seule fois.", is_error=True)
                after = before.replace(old, text)
            elif action == "reecrire":
                if not text:
                    return ToolResult("« texte » est vide : un fichier ne peut pas être vidé ainsi.", is_error=True)
                after = text
            else:
                return ToolResult(f"Commande inconnue : {action!r}.", is_error=True)

            # Lignes vides laissées par une suppression : retirées.
            after = "\n".join(line.rstrip() for line in after.splitlines() if line.strip())
            if len(after) > limit:
                logger.info("Mémoire : %s refusé sur %s (%s), %d caractères pour %d au plus",
                            action, name, label, len(after), limit)
                return ToolResult(
                    f"Refusé : {name} ferait {len(after)} caractères pour {limit} au plus. Réécris tout le "
                    f"fichier en plus court avec « reecrire », en fusionnant les lignes et en retirant le moins "
                    f"important, puis ajoute la nouvelle information.", is_error=True)
            if after == before:
                return ToolResult(f"Aucun changement : {name} contenait déjà cela.")
            try:
                await asyncio.to_thread(self._write_atomic, self.root / name, after)
            except OSError as exc:
                logger.error("Mémoire : impossible d'écrire %s : %s", name, exc)
                return ToolResult("Écriture impossible pour l'instant, à cause d'une erreur du bot.", is_error=True)
        logger.info("Mémoire : %s sur %s (%s), %d → %d caractères", action, name, label, len(before), len(after))
        return ToolResult(f"C'est fait. {name} : {len(after)}/{limit} caractères.", written=True)

    # ------------------------------------------------------------ internes

    @staticmethod
    def _normalize(name: str) -> str:
        """Accepte aussi « /memories/membres/123.md » ou « membres/123 » pour « membres/123.md »."""
        name = name.strip().replace("\\", "/").removeprefix("/memories/").lstrip("/")
        if name.startswith("membres/") and not name.endswith(".md"):
            name += ".md"
        return name

    def _check(self, name: str) -> tuple[int | None, str]:
        """Taille maximale du fichier et nom lisible pour les logs, ou (None, raison du refus).

        Seuls deux chemins existent : aucun autre fichier ne peut être lu ou écrit."""
        if name == SERVER_FILE:
            return self.server_limit, "fichier général"
        match = MEMBER_FILE.match(name)
        if match is None:
            return None, "Fichier inconnu : seuls serveur.md et membres/<identifiant>.md existent."
        entry = self.members.get(match.group(1))
        if entry is None:
            return None, "Membre inconnu : utilise un chemin de la liste des membres."
        if not entry["memory"]:
            return None, f"{display_name(entry)} a désactivé la mémoire : ne note rien sur ce membre."
        return self.member_limit, display_name(entry)

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        # Écrit dans un fichier temporaire puis le renomme : jamais de fichier à moitié écrit.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)
