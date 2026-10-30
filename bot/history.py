"""Mémoire des conversations par salon, sauvegardée dans un fichier JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
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
        # Marge au-delà de max_length avant de couper (voir _trim) : la moitié, en échanges entiers.
        self.margin = max(2, max_length // 4 * 2)
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
        """Garde au moins les N derniers messages, en commençant toujours par un message utilisateur.

        La mémoire est coupée par blocs : elle grandit jusqu'à N + marge messages, puis revient à N
        d'un coup. Entre deux coupes, son début ne change pas et reste dans le cache de prompt ;
        couper un échange à chaque message obligerait à réécrire tout l'historique en cache."""
        if len(messages) <= self.max_length + self.margin:
            trimmed = messages
        else:
            trimmed = messages[-self.max_length:]
        while trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]
        return trimmed

    def _load(self) -> dict[str, list[dict]]:
        if not self.file_path.exists():
            return {}
        try:
            channels = self._channels(json.loads(self.file_path.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:  # JSON invalide, mauvais encodage ou structure inattendue
            self._set_aside(exc)
            return {}
        return {str(cid): self._trim(self._sanitize(msgs)) for cid, msgs in channels.items()}

    @staticmethod
    def _channels(raw) -> dict:
        """Messages de chaque salon, quel que soit le format du fichier (ValueError s'il est inattendu)."""
        if not isinstance(raw, dict):
            raise ValueError(f"le fichier contient un {type(raw).__name__} au lieu d'un dictionnaire")
        if raw.get("version") == FILE_VERSION:
            channels = raw.get("channels") or {}
            if not isinstance(channels, dict):
                raise ValueError(f"« channels » est un {type(channels).__name__} au lieu d'un dictionnaire")
            return channels
        logger.info("Ancien format d'historique détecté : conversion automatique.")
        return {cid: val.get("messages") for cid, val in raw.items() if isinstance(val, dict)}

    def _set_aside(self, exc: Exception) -> None:
        """Met de côté un fichier illisible, sous un nom daté : il sera remplacé à la prochaine
        sauvegarde, et les copies précédentes ne sont jamais écrasées."""
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup = self.file_path.with_name(f"{self.file_path.name}.corrompu-{stamp}")
        logger.error("Fichier d'historique illisible (%s). Copie de secours : %s", exc, backup)
        try:
            os.replace(self.file_path, backup)
        except OSError as err:
            logger.error("Copie de secours impossible : %s", err)

    @staticmethod
    def _sanitize(messages) -> list[dict]:
        """Convertit des messages (éventuellement de l'ancien format Mistral) en texte simple,
        et fusionne les messages consécutifs d'un même rôle."""
        if not isinstance(messages, list):
            return []
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
