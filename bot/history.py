"""Mémoire des conversations par salon, sauvegardée dans un fichier JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from .text_utils import strip_reasoning

logger = logging.getLogger("chatbot.history")

FILE_VERSION = 3

# Remplace une image dans la mémoire (les images elles-mêmes ne sont pas conservées).
IMAGE_MARKER = "[image jointe : tu l'as vue et analysée à ce moment-là, mais elle n'est plus conservée en mémoire]"
_OLD_IMAGE_MARKER = "[image jointe]"
# Ajouté au message de l'utilisateur quand Claude a fait une recherche web pour y répondre
# (les recherches et leurs résultats ne sont pas conservés).
WEB_MARKER = "[recherche web : tu as réellement cherché sur Internet pour répondre à ce message, mais les résultats ne sont plus conservés en mémoire]"
# Ajouté à un message modifié sur Discord après avoir été mémorisé.
EDITED_MARKER = "[message modifié après ta réponse]"
# Remplace le texte d'un message supprimé de Discord après avoir été mémorisé…
DELETED_MARKER = "[message supprimé du salon depuis ta réponse]"
# … et sa citation dans les messages qui y répondaient.
DELETED_REPLY = "en réponse à un message supprimé depuis"
# Ce qu'un message supprimé perd dans la mémoire : seuls son identifiant, sa date et son auteur restent.
_DELETED_KEYS = ("reply_to", "reply", "forward", "content", "edited", "attachments")


def render_line(line: dict) -> str:
    """Texte d'une ligne de la mémoire, tel qu'il est envoyé à Claude.

    Une ligne est un message Discord (id, date, author, content ou forward, et éventuellement
    reply, reply_to, edited, attachments, ou deleted seul), ou une simple note du bot (content seul)."""
    if "author" not in line:
        return line["content"]
    head = f"{line['date']}, {line['author']}"
    if line.get("deleted"):
        return f"{head} a écrit : {DELETED_MARKER}"
    if "forward" in line:
        text = f"{head} a transféré un message {line['forward']}"
    else:
        reply = f" ({line['reply']})" if line.get("reply") else ""
        text = f"{head} a écrit{reply} : {line['content']}"
        if line.get("edited"):
            text += f" {EDITED_MARKER}"
    return " ".join([text, *line.get("attachments", [])])


class ConversationHistory:
    """Garde les derniers échanges de chaque salon.

    Chaque message de l'utilisateur garde ses lignes séparées, une par message Discord, avec
    son identifiant : un message modifié ou supprimé sur Discord l'est aussi dans la mémoire.
    Tout est en texte : une image est remplacée par IMAGE_MARKER, ce qui évite de conserver
    des liens Discord qui expirent et de repayer l'image à chaque message.
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
        """Renvoie les messages mémorisés pour ce salon, au format de l'API d'Anthropic."""
        return [self._render(m) for m in self._data.get(str(channel_id), [])]

    async def add_exchange(self, channel_id: int | str, lines: list[dict], assistant_text: str,
                           web_used: bool = False) -> None:
        """Ajoute un échange complet (lignes de l'utilisateur + réponse) puis sauvegarde."""
        key = str(channel_id)
        messages = self._data.setdefault(key, [])
        user: dict = {"role": "user", "lines": lines}
        if web_used:
            user["web"] = True
        messages.append(user)
        messages.append({"role": "assistant", "content": assistant_text})
        self._data[key] = self._trim(messages)
        await self._save()

    async def reset(self, channel_id: int | str) -> None:
        """Vide la mémoire du salon, après la fin d'une éventuelle réponse en cours."""
        # Sans le verrou, une réponse en cours réécrirait son échange juste après le reset.
        async with self.lock(channel_id):
            self._data[str(channel_id)] = []
            await self._save()

    async def edit(self, channel_id: int | str, message_id: int, content: str) -> bool:
        """Remplace le texte d'un message modifié sur Discord. Renvoie True s'il était mémorisé."""
        if not self._known(channel_id):
            return False
        # Le verrou attend la fin d'une réponse en cours : le message n'est mémorisé qu'après.
        async with self.lock(channel_id):
            for line in self._lines(channel_id):
                if line.get("id") == message_id and "content" in line and line["content"] != content:
                    line["content"] = content
                    line["edited"] = True
                    await self._save()
                    return True
        return False

    async def delete(self, channel_id: int | str, message_ids: set[int]) -> bool:
        """Efface le contenu des messages supprimés sur Discord, ainsi que leurs citations.

        La ligne reste, réduite à sa date et à son auteur, avec la réponse du bot : la
        conversation garde son fil sans conserver ce qui a été supprimé.
        Renvoie True si la mémoire a changé."""
        if not self._known(channel_id):
            return False
        async with self.lock(channel_id):
            changed = False
            for line in self._lines(channel_id):
                if line.get("id") in message_ids and not line.get("deleted"):
                    for key in _DELETED_KEYS:
                        line.pop(key, None)
                    line["deleted"] = True
                    changed = True
                elif line.get("reply_to") in message_ids:
                    line["reply"] = DELETED_REPLY
                    del line["reply_to"]
                    changed = True
            if changed:
                await self._save()
            return changed

    # ------------------------------------------------------------ internes

    def _known(self, channel_id: int | str) -> bool:
        """Salon dont la mémoire existe ou est en train d'être écrite."""
        return str(channel_id) in self._data or str(channel_id) in self._locks

    def _lines(self, channel_id: int | str):
        for entry in self._data.get(str(channel_id), []):
            if entry["role"] == "user":
                yield from entry["lines"]

    @staticmethod
    def _render(entry: dict) -> dict:
        if entry["role"] == "assistant":
            return {"role": "assistant", "content": entry["content"]}
        content = "\n".join(render_line(line) for line in entry["lines"])
        if entry.get("web"):
            content += f" {WEB_MARKER}"
        return {"role": "user", "content": content}

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
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
            channels = self._channels(raw)
        except (ValueError, OSError) as exc:  # JSON invalide, mauvais encodage ou structure inattendue
            self._set_aside(exc)
            return {}
        if raw.get("version") != FILE_VERSION:
            self._keep_old_copy(raw.get("version") or 1)
        return {str(cid): self._trim(self._sanitize(msgs)) for cid, msgs in channels.items()}

    @staticmethod
    def _channels(raw) -> dict:
        """Messages de chaque salon, quel que soit le format du fichier (ValueError s'il est inattendu)."""
        if not isinstance(raw, dict):
            raise ValueError(f"le fichier contient un {type(raw).__name__} au lieu d'un dictionnaire")
        if raw.get("version") in (2, FILE_VERSION):
            channels = raw.get("channels") or {}
            if not isinstance(channels, dict):
                raise ValueError(f"« channels » est un {type(channels).__name__} au lieu d'un dictionnaire")
            return channels
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

    def _keep_old_copy(self, version: int) -> None:
        """Copie un fichier d'un ancien format avant sa conversion : une version précédente
        du bot ne sait pas lire le nouveau format et repartirait d'une mémoire vide."""
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup = self.file_path.with_name(f"{self.file_path.name}.v{version}-{stamp}")
        try:
            shutil.copy2(self.file_path, backup)
        except OSError as err:
            logger.error("Copie de l'ancien historique impossible : %s", err)
            return
        logger.info("Historique au format %s : conversion au format %s. Copie de l'ancien fichier : %s",
                    version, FILE_VERSION, backup)

    @classmethod
    def _sanitize(cls, messages) -> list[dict]:
        """Met les messages au format actuel (en convertissant les anciens formats, dont celui
        de Mistral), et fusionne les messages consécutifs d'un même rôle."""
        if not isinstance(messages, list):
            return []
        clean: list[dict] = []
        for msg in messages:
            entry = cls._entry(msg)
            if entry is None:
                continue
            if clean and clean[-1]["role"] == entry["role"]:
                previous = clean[-1]
                if entry["role"] == "user":
                    previous["lines"] += entry["lines"]
                    if entry.get("web"):
                        previous["web"] = True
                else:
                    previous["content"] += "\n" + entry["content"]
            else:
                clean.append(entry)
        return clean

    @classmethod
    def _entry(cls, msg) -> dict | None:
        """Un message du fichier au format actuel, ou None s'il est vide ou illisible."""
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            return None
        if msg["role"] == "user" and isinstance(msg.get("lines"), list):
            lines = [line for line in msg["lines"] if cls._readable(line)]
            if not lines:
                return None
            return {"role": "user", "lines": lines, **({"web": True} if msg.get("web") else {})}

        # Anciens formats : un seul texte par message.
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
            return None
        if msg["role"] == "assistant":
            # Réflexion affichée par erreur avant la 5.8 : le modèle l'imiterait en la relisant.
            content = strip_reasoning(content)
            return {"role": "assistant", "content": content} if content else None
        # Le texte converti reste identique une fois rendu : le cache de prompt est conservé.
        web = content.endswith(f" {WEB_MARKER}")
        if web:
            content = content[: -len(WEB_MARKER) - 1]
        if content.endswith(_OLD_IMAGE_MARKER):
            content = content[: -len(_OLD_IMAGE_MARKER)] + IMAGE_MARKER
        return {"role": "user", "lines": [{"content": content}], **({"web": True} if web else {})}

    @staticmethod
    def _readable(line) -> bool:
        if not isinstance(line, dict):
            return False
        try:
            return bool(render_line(line).strip())
        except (KeyError, TypeError, AttributeError):
            return False

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
