"""Mémoire des conversations par salon, sauvegardée dans un fichier JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from .files import write_atomic

logger = logging.getLogger("chatbot.history")

FILE_VERSION = 3

# Durée de vie du cache de prompt sans nouveau message (en secondes).
CACHE_LIFETIME = 5 * 60

# Remplace une image dans la mémoire (les images elles-mêmes ne sont pas conservées).
IMAGE_MARKER = "[image jointe : tu l'as vue et analysée à ce moment-là, mais elle n'est plus conservée en mémoire]"
# Ajouté au message de l'utilisateur quand Claude a fait une recherche web pour y répondre
# (les recherches et leurs résultats ne sont pas conservés).
WEB_MARKER = "[recherche web : tu as réellement cherché sur Internet pour répondre à ce message, mais les résultats ne sont plus conservés en mémoire]"
# Ajouté à un message modifié sur Discord après avoir été mémorisé.
EDITED_MARKER = "[message modifié après ta réponse]"
# Remplace le texte d'un message supprimé de Discord après avoir été mémorisé…
DELETED_MARKER = "[message supprimé du salon depuis ta réponse]"
# … et sa citation dans les messages qui y répondaient.
DELETED_REPLY = "en réponse à un message supprimé depuis"
# Remplace une réponse du bot supprimée du salon (par lui-même ou par un modérateur), ou seulement le passage
# supprimé quand sa réponse tenait en plusieurs messages Discord.
OWN_DELETED_MARKER = "[ton message supprimé du salon]"
# Suit le texte d'un message que le bot a modifié lui-même, après l'avoir envoyé.
OWN_EDITED_MARKER = "[tu as modifié ce message après l'avoir envoyé]"
# Ce qu'un message supprimé perd dans la mémoire : seuls son identifiant, sa date et son auteur restent.
_DELETED_KEYS = ("reply_to", "reply", "forward", "content", "edited", "attachments")


def images_marker(count: int) -> str:
    """Remplace les images d'un message dans la mémoire : un seul repère, quel que soit leur nombre."""
    if count == 1:
        return IMAGE_MARKER
    return f"[{count} images jointes : tu les as vues et analysées à ce moment-là, mais elles ne sont plus conservées en mémoire]"


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
        # Taille au-delà de laquelle la mémoire revient à max_length, quand les messages s'enchaînent (voir _trim).
        self.limit = 3 * max_length
        self.clock = time.monotonic
        self._data: dict[str, list[dict]] = self._load()
        self._last_sent: dict[str, float] = {}  # salon : heure (clock) du dernier envoi de la mémoire à Claude
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
        """Renvoie les messages mémorisés pour ce salon, au format de l'API d'Anthropic, juste avant de les
        envoyer à Claude.

        Après une pause qui a vidé le cache de prompt, la mémoire revient d'abord à max_length messages :
        elle sera réécrite en cache en entier de toute façon, autant qu'elle soit courte. Le cache démarre
        quand la question part chez Claude : c'est donc ici que l'heure est notée, pas à la réponse."""
        key = str(channel_id)
        now = self.clock()
        last = self._last_sent.get(key)
        if key in self._data and (last is None or now - last > CACHE_LIFETIME):
            self._data[key] = self._cut(self._data[key])
        self._last_sent[key] = now
        return [self._render(m) for m in self._data.get(key, [])]

    async def add_exchange(self, channel_id: int | str, lines: list[dict], assistant_text: str,
                           web_used: bool = False, reply_ids: list[int] | None = None) -> dict:
        """Ajoute un échange complet (lignes de l'utilisateur + réponse) puis sauvegarde.

        reply_ids : identifiants des messages Discord de la réponse, si elle est déjà envoyée. Sinon, les donner
        ensuite à set_reply_ids avec l'entrée de la réponse, que cette méthode renvoie."""
        key = str(channel_id)
        messages = self._data.setdefault(key, [])
        user: dict = {"role": "user", "lines": lines}
        if web_used:
            user["web"] = True
        messages.append(user)
        reply: dict = {"role": "assistant", "content": assistant_text}
        if reply_ids:
            reply["ids"] = list(reply_ids)
        messages.append(reply)
        self._data[key] = self._trim(messages)
        await self._save()
        return reply

    async def set_reply_ids(self, reply: dict, reply_ids: list[int]) -> None:
        """Note les identifiants des messages Discord d'une réponse envoyée après avoir été mémorisée : sa
        suppression du salon pourra être retrouvée (voir delete)."""
        if reply_ids:
            reply["ids"] = list(reply_ids)
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

    async def delete(self, channel_id: int | str, message_ids: set[int],
                     texts: dict[int, str] | None = None) -> bool:
        """Efface le contenu des messages supprimés sur Discord, ainsi que leurs citations.

        La ligne d'un membre reste, réduite à sa date et à son auteur, avec la réponse du bot : la
        conversation garde son fil sans conserver ce qui a été supprimé. Une réponse du bot supprimée devient
        OWN_DELETED_MARKER (voir _forget_reply ; texts : texte des messages supprimés, quand il est connu).
        Renvoie True si la mémoire a changé."""
        if not self._known(channel_id):
            return False
        async with self.lock(channel_id):
            changed = False
            for entry in self._data.get(str(channel_id), []):
                if entry["role"] == "assistant" and message_ids & set(entry.get("ids", ())):
                    self._forget_reply(entry, message_ids, texts or {})
                    changed = True
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

    @staticmethod
    def _forget_reply(entry: dict, message_ids: set[int], texts: dict[int, str]) -> None:
        """Retire d'une réponse mémorisée ses messages Discord supprimés.

        Une réponse en plusieurs messages perd seulement le passage supprimé, s'il se retrouve d'après le texte
        du message (connu tant que Discord le garde en cache : le cas d'un message récent). Sinon, toute la
        réponse est oubliée : mieux vaut perdre un passage resté visible que garder celui qu'un modérateur a retiré."""
        deleted = [i for i in entry["ids"] if i in message_ids]
        remaining = [i for i in entry["ids"] if i not in message_ids]
        content = entry["content"]
        for message_id in deleted if remaining else ():
            text = texts.get(message_id, "").strip()
            if not text or text not in content:
                remaining = []
                break
            edited = f"{text} {OWN_EDITED_MARKER}"  # le repère d'une modification part avec le message
            content = content.replace(edited if edited in content else text, OWN_DELETED_MARKER, 1)
        if remaining:
            entry["content"], entry["ids"] = content, remaining
        else:
            entry["content"] = OWN_DELETED_MARKER
            del entry["ids"]

    async def revise_reply(self, channel_id: int | str, message_id: int, before: str, after: str) -> bool:
        """Le bot a modifié l'un de ses messages Discord (outil « mes_messages ») : before, son texte complet
        avant, devient after dans la réponse mémorisée, suivi de OWN_EDITED_MARKER. Sans ce repère, le bot
        verrait son message déjà modifié, sans trace de son geste, et croirait n'y avoir jamais touché.

        La réponse se retrouve par l'identifiant du message ; pour une réponse mémorisée sans identifiants
        (avant la 5.18), c'est la plus récente qui contient before. Renvoie True si la mémoire a changé.
        Sans verrou : appelé pendant une réponse, qui tient déjà celui du salon."""
        replies = [entry for entry in self._data.get(str(channel_id), []) if entry["role"] == "assistant"]
        entry = next((e for e in replies if message_id in e.get("ids", ())), None)
        if entry is None:
            entry = next((e for e in reversed(replies) if "ids" not in e and before in e["content"]), None)
        if entry is None or not after.strip():  # l'API refuse un message vide
            return False
        content = entry["content"]
        if before in content:
            start = content.index(before)
            end = start + len(after)
            content = content[:start] + after + content[start + len(before):]
            if not content[end:].startswith(f" {OWN_EDITED_MARKER}"):  # déjà là après une modification précédente
                content = f"{content[:end]} {OWN_EDITED_MARKER}{content[end:]}"
        elif entry.get("ids") == [message_id]:
            content = f"{after} {OWN_EDITED_MARKER}"  # la réponse entière était ce message
        else:
            # Passage introuvable dans une réponse en plusieurs messages (texte mémorisé différent de celui
            # envoyé, par exemple là où un long code est parti en fichier joint) : au moins, le bot le sait.
            logger.warning("Message %s modifié, introuvable dans la mémoire du salon %s : seul le repère est ajouté.",
                           message_id, channel_id)
            if not content.endswith(OWN_EDITED_MARKER):
                content = f"{content} {OWN_EDITED_MARKER}"
        entry["content"] = content
        await self._save()
        return True

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

        La mémoire est coupée par blocs : elle grandit jusqu'à 3 × N messages, puis revient à N d'un coup.
        Chaque coupe change le début de la conversation et oblige à réécrire tout l'historique en cache,
        ce qui coûte bien plus cher que d'en relire une partie plus longue (25 fois plus sur Opus 5.5).
        Après une pause, get() la ramène à N sans attendre."""
        return self._cut(messages) if len(messages) > self.limit else self._from_user(messages)

    def _cut(self, messages: list[dict]) -> list[dict]:
        """Les N derniers messages, en commençant par un message utilisateur."""
        return self._from_user(messages[-self.max_length:])

    @staticmethod
    def _from_user(messages: list[dict]) -> list[dict]:
        trimmed = messages
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
        return {str(cid): self._trim(self._sanitize(msgs)) for cid, msgs in channels.items()}

    @staticmethod
    def _channels(raw) -> dict:
        """Messages de chaque salon (ValueError si le fichier est inattendu)."""
        if not isinstance(raw, dict):
            raise ValueError(f"le fichier contient un {type(raw).__name__} au lieu d'un dictionnaire")
        if raw.get("version") != FILE_VERSION:
            raise ValueError(f"format {raw.get('version')!r} inconnu (attendu : {FILE_VERSION})")
        channels = raw.get("channels") or {}
        if not isinstance(channels, dict):
            raise ValueError(f"« channels » est un {type(channels).__name__} au lieu d'un dictionnaire")
        return channels

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

    @classmethod
    def _sanitize(cls, messages) -> list[dict]:
        """Garde les messages lisibles, et fusionne les messages consécutifs d'un même rôle (laissés
        côte à côte par un message illisible retiré entre eux)."""
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
                    if entry.get("ids"):
                        previous["ids"] = previous.get("ids", []) + entry["ids"]
            else:
                clean.append(entry)
        return clean

    @classmethod
    def _entry(cls, msg) -> dict | None:
        """Un message du fichier, ou None s'il est vide ou illisible."""
        if not isinstance(msg, dict):
            return None
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                return None
            ids = msg.get("ids")
            if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
                return {"role": "assistant", "content": content, "ids": ids}
            return {"role": "assistant", "content": content}
        if msg.get("role") == "user" and isinstance(msg.get("lines"), list):
            lines = [line for line in msg["lines"] if cls._readable(line)]
            if lines:
                return {"role": "user", "lines": lines, **({"web": True} if msg.get("web") else {})}
        return None

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
                await asyncio.to_thread(write_atomic, self.file_path, text)
            except OSError as exc:
                logger.error("Impossible de sauvegarder l'historique : %s", exc)
