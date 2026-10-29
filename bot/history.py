"""Mémoire des conversations par salon, sauvegardée dans un fichier JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("chatbot.history")

FILE_VERSION = 2

# Remplace une image dans la mémoire (les images elles-mêmes ne sont pas conservées).
IMAGE_MARKER = "[image jointe : tu l'as vue et analysée à ce moment-là, mais elle n'est plus conservée en mémoire]"
_OLD_IMAGE_MARKER = "[image jointe]"
# Ajouté au message de l'utilisateur quand Claude a fait une recherche web pour y répondre
# (les recherches et leurs résultats ne sont pas conservés).
WEB_MARKER = "[recherche web : tu as réellement cherché sur Internet pour répondre à ce message, mais les résultats ne sont plus conservés en mémoire]"


class ConversationHistory:
    """Garde les derniers échanges de chaque salon.

    Les messages sont stockés au format de l'API d'Anthropic, en texte uniquement :
    une image est remplacée par IMAGE_MARKER, ce qui évite de conserver des liens
    Discord qui expirent et de repayer l'image à chaque message.
    """

    def __init__(self, file_path: Path, max_length: int):
        self.file_path = file_path
        self.max_length = max_length
        self._data: dict[str, list[dict]] = self._load()
        self._locks: dict[str, asyncio.Lock] = {}
        self._save_lock = asyncio.Lock()

    # ------------------------------------------------------------------ accès

    def lock(self, channel_id: int | str) -> asyncio.Lock:
        """Verrou par salon : les messages d'un même salon sont traités un par un."""
        key = str(channel_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def get(self, channel_id: int | str) -> list[dict]:
        """Renvoie une copie des messages mémorisés pour ce salon."""
        return [dict(m) for m in self._data.get(str(channel_id), [])]

    async def add_exchange(self, channel_id: int | str, user_text: str, assistant_text: str) -> None:
        """Ajoute un échange complet (question + réponse) puis sauvegarde."""
        key = str(channel_id)
        messages = self._data.setdefault(key, [])
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
        self._data[key] = self._trim(messages)
        await self._save()

    async def reset(self, channel_id: int | str) -> None:
        """Vide la mémoire du salon, après la fin d'une éventuelle réponse en cours."""
        # Sans le verrou, une réponse en cours réécrirait son échange juste après le reset.
        async with self.lock(channel_id):
            self._data[str(channel_id)] = []
            await self._save()

    # ------------------------------------------------------------ internes

    def _trim(self, messages: list[dict]) -> list[dict]:
        """Garde les N derniers messages, en commençant toujours par un message utilisateur."""
        trimmed = messages[-self.max_length:]
        while trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]
        return trimmed

    def _load(self) -> dict[str, list[dict]]:
        if not self.file_path.exists():
            return {}
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            backup = self.file_path.with_suffix(self.file_path.suffix + ".corrompu")
            logger.error("Fichier d'historique illisible (%s). Copie de secours : %s", exc, backup)
            try:
                os.replace(self.file_path, backup)
            except OSError:
                pass
            return {}

        if isinstance(raw, dict) and raw.get("version") == FILE_VERSION:
            channels = raw.get("channels", {})
        else:
            logger.info("Ancien format d'historique détecté : conversion automatique.")
            channels = {cid: (val or {}).get("messages", []) for cid, val in (raw or {}).items()
                        if isinstance(val, dict)}

        return {str(cid): self._trim(self._sanitize(msgs)) for cid, msgs in channels.items()}

    @staticmethod
    def _sanitize(messages: list) -> list[dict]:
        """Convertit des messages (éventuellement de l'ancien format Mistral) en texte simple,
        et fusionne les messages consécutifs d'un même rôle."""
        clean: list[dict] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") in ("image", "image_url"):
                        parts.append(IMAGE_MARKER)
                content = " ".join(p for p in parts if p)
            if not isinstance(content, str) or not content.strip():
                continue
            if msg["role"] == "user" and content.endswith(_OLD_IMAGE_MARKER):
                content = content[: -len(_OLD_IMAGE_MARKER)] + IMAGE_MARKER
            if clean and clean[-1]["role"] == msg["role"]:
                clean[-1]["content"] += "\n" + content
            else:
                clean.append({"role": msg["role"], "content": content})
        return clean

    async def _save(self) -> None:
        payload = {"version": FILE_VERSION, "channels": self._data}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        async with self._save_lock:
            try:
                await asyncio.to_thread(self._write_atomic, text)
            except OSError as exc:
                logger.error("Impossible de sauvegarder l'historique : %s", exc)

    def _write_atomic(self, text: str) -> None:
        # Écrit dans un fichier temporaire puis le renomme : jamais de fichier à moitié écrit.
        tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.file_path)
