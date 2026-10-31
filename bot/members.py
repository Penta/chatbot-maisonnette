"""Registre des membres vus par le bot (noms, anciens pseudos, réglages de la mémoire), dans un fichier JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("chatbot.members")

FILE_VERSION = 1
# Noms Discord d'un membre : nom d'utilisateur (unique), nom d'affichage, pseudo sur le serveur.
NAME_KEYS = ("username", "global_name", "nickname")
# Anciens pseudos gardés par membre, du plus ancien au plus récent.
MAX_FORMER_NAMES = 5


class MemberRegistry:
    """Garde, pour chaque membre vu (auteur d'un message lu par le bot, ou membre mentionné), ses noms
    actuels et ses anciens pseudos, s'il a été prévenu de la mémoire et s'il l'a désactivée.

    Le fichier n'est réécrit que si quelque chose change : pas d'écriture à chaque message."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data: dict[str, dict] = self._load()
        self._save_lock = asyncio.Lock()

    # ------------------------------------------------------------------ accès

    def get(self, member_id: int | str) -> dict | None:
        return self._data.get(str(member_id))

    def members(self) -> dict[str, dict]:
        """Tous les membres connus, par identifiant."""
        return self._data

    def memory_enabled(self, member_id: int | str) -> bool:
        """La mémoire est activée par défaut, y compris pour un membre encore inconnu."""
        entry = self.get(member_id)
        return entry is None or entry["memory"]

    async def see(self, member_id: int, username: str, global_name: str | None,
                  nickname: str | None = None, in_guild: bool = True) -> None:
        """Enregistre un membre ou met ses noms à jour ; un nom remplacé rejoint les anciens pseudos.

        in_guild=False (message privé) : le pseudo sur le serveur n'est pas connu, il reste inchangé."""
        names = {"username": username, "global_name": global_name or None}
        if in_guild:
            names["nickname"] = nickname or None
        key = str(member_id)
        entry = self._data.get(key)
        if entry is None:
            self._data[key] = {
                **{name_key: None for name_key in NAME_KEYS}, **names,
                "former_names": [], "notified": False, "memory": True,
                "first_seen": datetime.now().strftime("%Y-%m-%d"),
            }
            logger.info("Nouveau membre dans le registre : %s (%s)", display_name(self._data[key]), key)
            await self._save()
            return
        old_names = [entry[name_key] for name_key in NAME_KEYS if entry[name_key]]
        changed = False
        for name_key, value in names.items():
            if entry[name_key] != value:
                entry[name_key] = value
                changed = True
        if not changed:
            return
        current = {entry[name_key] for name_key in NAME_KEYS if entry[name_key]}
        former = [name for name in entry["former_names"] if name not in current]
        for name in old_names:
            if name not in current and name not in former:
                former.append(name)
        entry["former_names"] = former[-MAX_FORMER_NAMES:]
        logger.info("Noms mis à jour dans le registre : %s (%s)", display_name(entry), key)
        await self._save()

    async def set_notified(self, member_id: int | str) -> None:
        """Le membre a été prévenu de la mémoire : il ne le sera plus."""
        if (entry := self.get(member_id)) is not None and not entry["notified"]:
            entry["notified"] = True
            await self._save()

    async def set_memory(self, member_id: int | str, enabled: bool) -> None:
        """Active ou désactive la mémoire pour ce membre."""
        if (entry := self.get(member_id)) is not None and entry["memory"] != enabled:
            entry["memory"] = enabled
            logger.info("Mémoire %s pour %s (%s)", "activée" if enabled else "désactivée", display_name(entry), member_id)
            await self._save()

    # ------------------------------------------------------------ internes

    def _load(self) -> dict[str, dict]:
        if not self.file_path.exists():
            return {}
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("members"), dict):
                raise ValueError("« members » absent ou n'est pas un dictionnaire")
        except (ValueError, OSError) as exc:  # JSON invalide, mauvais encodage ou structure inattendue
            self._set_aside(exc)
            return {}
        return {key: entry for key, value in raw["members"].items() if (entry := self._sanitize(value))}

    @staticmethod
    def _sanitize(value) -> dict | None:
        """Une entrée du fichier, complétée si besoin, ou None si elle est illisible."""
        if not isinstance(value, dict) or not isinstance(value.get("username"), str):
            return None
        entry = {name_key: value.get(name_key) if isinstance(value.get(name_key), str) else None for name_key in NAME_KEYS}
        former = value.get("former_names")
        entry["former_names"] = [n for n in former if isinstance(n, str)] if isinstance(former, list) else []
        entry["notified"] = value.get("notified") is True
        entry["memory"] = value.get("memory") is not False
        entry["first_seen"] = value.get("first_seen") if isinstance(value.get("first_seen"), str) else None
        return entry

    def _set_aside(self, exc: Exception) -> None:
        """Met de côté un fichier illisible, sous un nom daté, comme pour l'historique."""
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup = self.file_path.with_name(f"{self.file_path.name}.corrompu-{stamp}")
        logger.error("Registre des membres illisible (%s). Copie de secours : %s", exc, backup)
        try:
            os.replace(self.file_path, backup)
        except OSError as err:
            logger.error("Copie de secours impossible : %s", err)

    async def _save(self) -> None:
        payload = {"version": FILE_VERSION, "members": self._data}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        async with self._save_lock:
            try:
                await asyncio.to_thread(self._write_atomic, text)
            except OSError as exc:
                logger.error("Impossible de sauvegarder le registre des membres : %s", exc)

    def _write_atomic(self, text: str) -> None:
        # Écrit dans un fichier temporaire puis le renomme : jamais de fichier à moitié écrit.
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.file_path)


def display_name(entry: dict) -> str:
    """Le nom affiché sur le serveur : pseudo, sinon nom d'affichage, sinon nom d'utilisateur."""
    return entry["nickname"] or entry["global_name"] or entry["username"]
