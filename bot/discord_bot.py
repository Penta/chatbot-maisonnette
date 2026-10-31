"""Le bot Discord : événements, commandes et réactions."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import discord
from discord import app_commands
from discord.ext import commands

from . import VERSION
from .claude_client import ClaudeClient, ClaudeError, ImageData
from .config import Settings
from .emojis import discord_emojis, emoji_names, short_emojis, usable_emojis
from .history import IMAGE_MARKER, ConversationHistory, render_line
from .images import ImageError, prepare_image
from .pdf_files import PDF_EXTENSIONS, PDF_MAX_BYTES, PdfError, prepare_pdf
from .presence import PresenceManager, Status
from .text_files import TEXT_EXTENSIONS, TEXT_FILE_MAX_BYTES, TextFileError, decode_text_file
from .text_utils import split_message

logger = logging.getLogger("chatbot.discord")

CUSTOM_EMOJI_ONLY = re.compile(r"^<a?:\w+:\d+>$")
USER_MENTION = re.compile(r"<@!?(\d+)>")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff")
# Longueur maximale du texte d'un message cité ou transféré transmis à Claude.
QUOTE_MAX_LENGTH = 500
# Messages écrits par quelqu'un ; les autres types sont des messages système
# (arrivée d'un membre, message épinglé, boost, fil créé…).
USER_MESSAGE_TYPES = (discord.MessageType.default, discord.MessageType.reply)
# En passant absent ou en « Ne pas déranger », le bot prévient son salon si on y a parlé depuis moins de :
DEPARTURE_WINDOW = timedelta(minutes=3)


def looks_like_image(attachment: discord.Attachment) -> bool:
    return (attachment.content_type or "").startswith("image/") or attachment.filename.lower().endswith(IMAGE_EXTENSIONS)


def looks_like_text(attachment: discord.Attachment) -> bool:
    return attachment.filename.lower().endswith(TEXT_EXTENSIONS)


def looks_like_pdf(attachment: discord.Attachment) -> bool:
    return attachment.content_type == "application/pdf" or attachment.filename.lower().endswith(PDF_EXTENSIONS)


@contextlib.asynccontextmanager
async def typing(channel: discord.abc.Messageable):
    """« En train d'écrire… » pendant le bloc, sans empêcher la réponse si Discord le refuse."""
    indicator = channel.typing()
    try:
        await indicator.__aenter__()  # seul ce premier envoi peut échouer : les suivants sont protégés
    except discord.HTTPException as exc:
        logger.warning("Indicateur « en train d'écrire » impossible dans #%s : %s", channel, exc)
        indicator = None
    try:
        yield
    finally:
        if indicator is not None:
            await indicator.__aexit__(None, None, None)


def text_file_prompt(name: str, content: str) -> str:
    """Contenu d'un fichier .txt tel qu'il est transmis à Claude."""
    return f"\n\n[Contenu du fichier joint « {name} » :]\n{content}\n[Fin du fichier « {name} »]"


def text_file_marker(name: str) -> str:
    """Remplace un fichier texte dans la mémoire (son contenu n'est pas conservé)."""
    return f"[fichier texte joint « {name} » : tu l'as lu à ce moment-là, mais son contenu n'est plus conservé en mémoire]"


def attachment_marker(attachment: ImageData) -> str:
    """Remplace une image ou un PDF dans la mémoire (son contenu n'est pas conservé)."""
    if attachment.is_pdf:
        return f"[PDF joint « {attachment.name} » : tu l'as lu à ce moment-là, mais son contenu n'est plus conservé en mémoire]"
    return IMAGE_MARKER


@dataclass
class PendingMessage:
    """Message du salon dédié, prêt à être envoyé à Claude."""
    message: discord.Message
    line: dict   # ligne conservée en mémoire (voir history.render_line)
    extra: str   # ajouté pour Claude seulement : contenu du fichier texte joint
    image: ImageData | None

    @property
    def user_text(self) -> str:
        """Texte envoyé à Claude : l'image ou le fichier lui-même remplace son repère."""
        line = {key: value for key, value in self.line.items() if key != "attachments"}
        return render_line(line) + self.extra


class ChatBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True  # nécessaire pour lire le texte des messages
        super().__init__(
            command_prefix=commands.when_mentioned,  # aucune commande textuelle utilisée
            intents=intents,
            help_command=None,
            # Le bot ne peut jamais pinger @everyone, @here, des rôles ou des membres.
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.claude = ClaudeClient(settings)
        self.history = ConversationHistory(settings.history_file, settings.max_history_length)
        self._pending: dict[int, list[PendingMessage]] = {}  # messages en attente de réponse, par salon
        self._announced = False
        self.presence = PresenceManager(self, self.claude, settings) if settings.presence else None
        # Absent ou en « Ne pas déranger » : réponses différées, par salon.
        self._waiters: dict[int, asyncio.Task] = {}  # attente en cours
        self._mentions: dict[int, list[discord.Message]] = {}  # mentions en attente (hors salon dédié)
        # Messages en attente, gardés sur le disque pour y répondre même après un redémarrage.
        self._waiting_file = settings.history_file.with_name("waiting_messages.json")
        self._waiting: set[tuple[int, int]] = set()  # (salon, message)
        self._last_main_message: datetime | None = None  # date du dernier message reçu dans le salon dédié
        # Empreinte des commandes slash envoyées à Discord (voir sync_commands).
        self._commands_file = settings.history_file.with_name("slash_commands.sha256")

    # ------------------------------------------------------------ cycle de vie

    async def setup_hook(self) -> None:
        # Exécuté une seule fois au démarrage (contrairement à on_ready).
        self.tree.add_command(reset_command)
        await self.sync_commands()
        s = self.settings
        logger.info("Modèle %s — réflexion : %s, effort : %s, recherche web : %s",
                    s.model, "on" if s.thinking else "off", s.effort or "par défaut",
                    "on" if s.web_search else "off")
        if self.presence:
            self.presence.start()

    async def sync_commands(self) -> None:
        """Envoie les commandes slash à Discord, seulement si elles ont changé depuis le dernier envoi
        (Discord limite ces envois : inutile de les refaire à chaque démarrage)."""
        payload = [command.to_dict(self.tree) for command in self.tree.get_commands()]
        signature = json.dumps([self.application_id, payload], sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        try:
            if self._commands_file.read_text(encoding="utf-8").strip() == digest:
                logger.info("Commandes slash inchangées : pas de synchronisation.")
                return
        except OSError:
            pass  # premier démarrage, ou fichier illisible : on synchronise
        try:
            synced = await self.tree.sync()
        except discord.HTTPException as exc:  # le bot fonctionne quand même ; nouvel essai au prochain démarrage
            logger.error("Synchronisation des commandes slash impossible : %s", exc)
            return
        logger.info("%d commande(s) slash synchronisée(s).", len(synced))
        try:
            self._commands_file.write_text(digest, encoding="utf-8")
        except OSError as exc:
            logger.warning("Impossible d'écrire %s : %s", self._commands_file, exc)

    async def on_ready(self) -> None:
        logger.info("Connecté en tant que %s (version %s)", self.user, VERSION)
        if self.presence:
            await self.presence.apply()  # une reconnexion complète efface le statut
        if self._announced:
            return  # simple reconnexion : on ne reposte pas le message de démarrage
        self._announced = True
        self._replay_task = asyncio.create_task(self.replay_waiting())  # référence gardée : pas de ramasse-miettes

        channel = self.get_channel(self.settings.channel_id)
        if channel is None:
            logger.error("Salon %s introuvable : vérifie CHANNEL_ID et les permissions du bot.",
                         self.settings.channel_id)
            return
        name = channel.guild.me.display_name if getattr(channel, "guild", None) else self.user.name
        logger.info("%d emoji(s) du serveur utilisable(s) dans les réponses.",
                    len(usable_emojis(getattr(channel, "guild", None))))
        embed = discord.Embed(
            title="Bot en ligne",
            description=f"{name} est désormais en ligne. Version {VERSION}.",
            color=discord.Color.green(),
        )
        try:
            await channel.send(embed=embed)
        except discord.HTTPException as exc:
            logger.error("Impossible d'envoyer le message de démarrage : %s", exc)

    async def close(self) -> None:
        if self.presence:
            self.presence.stop()
        for task in self._waiters.values():
            task.cancel()  # les messages en attente restent dans waiting_messages.json
        await self.claude.close()
        await super().close()

    # ------------------------------------------------------------- messages

    async def on_message(self, message: discord.Message) -> None:
        # Ignore ses propres messages et les webhooks : évite les boucles infinies.
        if message.author == self.user or message.webhook_id is not None:
            return
        if message.type not in USER_MESSAGE_TYPES:
            return  # message système
        # Les autres bots ne sont écoutés que lorsqu'ils s'adressent au bot : mention,
        # ou réponse à l'un de ses messages. Deux bots peuvent ainsi discuter ensemble.
        if message.author.bot and self.user not in message.mentions and not await self.replies_to_me(message):
            return

        in_main_channel = message.channel.id == self.settings.channel_id
        # Mention directe uniquement (@everyone et @here ne comptent pas).
        mentioned = self.user is not None and self.user in message.mentions

        if in_main_channel:
            await self.handle_main_channel(message)
        elif mentioned:
            await self.handle_mention(message)

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        """Message modifié : la mémoire, ou le message s'il attend encore sa réponse, prend le nouveau texte."""
        message = payload.message
        # Aperçu d'un lien qui s'affiche, ou transfert (qui ne se modifie pas) : rien n'a changé.
        if message.edited_at is None or message.message_snapshots:
            return
        content = self.message_body(message)
        for pending in self._pending.get(payload.channel_id, []):
            if pending.message.id == message.id:
                pending.line["content"] = content
        if await self.history.edit(payload.channel_id, message.id, content):
            logger.info("Message modifié dans le salon %s : mémoire mise à jour.", payload.channel_id)

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        await self.forget_deleted(payload.channel_id, {payload.message_id})

    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent) -> None:
        await self.forget_deleted(payload.channel_id, payload.message_ids)

    async def forget_deleted(self, channel_id: int, message_ids: set[int]) -> None:
        """Messages supprimés : ils sortent de la mémoire et n'attendent plus de réponse."""
        # Listes modifiées sur place : une réponse sur le point de partir les lit encore.
        if pending := self._pending.get(channel_id):
            pending[:] = [p for p in pending if p.message.id not in message_ids]
        if mentions := self._mentions.get(channel_id):
            mentions[:] = [m for m in mentions if m.id not in message_ids]
        keys = {(channel_id, message_id) for message_id in message_ids}
        if keys & self._waiting:
            self._waiting -= keys
            self._save_waiting()
        if await self.history.delete(channel_id, message_ids):
            logger.info("%s message(s) supprimé(s) dans le salon %s : mémoire mise à jour.", len(message_ids), channel_id)

    async def handle_main_channel(self, message: discord.Message) -> None:
        if self._last_main_message is None or message.created_at > self._last_main_message:
            self._last_main_message = message.created_at
        content = message.content.strip()
        # Un sticker ou un emoji seul reçoit un sticker ou un emoji du serveur. S'il n'y en a aucun
        # d'utilisable, Claude répond avec des mots ; accompagné de texte, le sticker lui est décrit.
        if (message.stickers and not content and not message.attachments) or CUSTOM_EMOJI_ONLY.match(content):
            if not self.available():
                return  # absent : pas de réaction aux stickers ni aux emojis
            if await self.reply_to_sticker(message) or await self.reply_to_emoji(message):
                return

        forward, forward_image = await self.read_forward(message)
        if not (content or message.attachments or message.stickers or forward):
            return  # sondage, transfert d'un message vide… : rien à lire

        attachments = await self.read_attachments(message)
        if attachments is False:
            return  # pièce jointe refusée, l'utilisateur a été prévenu
        image, text_file = attachments

        # L'image du message cité n'est utilisée que si le message n'en contient pas lui-même.
        reply, reply_image = await self.read_reply(message, want_image=image is None and forward_image is None)
        image = image or forward_image or reply_image

        line: dict = {"id": message.id, "date": self.timestamp(message), "author": message.author.display_name}
        if forward:
            line["forward"] = forward
        else:
            if reply:
                line["reply_to"] = message.reference.message_id  # citation retirée si ce message est supprimé
                line["reply"] = reply
            line["content"] = self.message_body(message)
        markers = [attachment_marker(image)] if image else []
        extra = ""
        if text_file:
            extra = text_file_prompt(*text_file)
            markers.append(text_file_marker(text_file[0]))
        if markers:
            line["attachments"] = markers

        # Un seul appel à la fois par salon : l'historique reste cohérent. Les messages arrivés
        # pendant une réponse attendent dans la file, puis partent ensemble dans un seul appel.
        self._pending.setdefault(message.channel.id, []).append(PendingMessage(message, line, extra, image))
        if not self.available() or message.channel.id in self._waiters:
            self.answer_later(message, self.answer_main_channel)
            return
        await self.answer_main_channel(message.channel)

    async def answer_main_channel(self, channel: discord.abc.Messageable, late: Status | None = None) -> None:
        """Répond en une fois aux messages en attente du salon dédié."""
        pending = self._pending.setdefault(channel.id, [])
        async with self.history.lock(channel.id):
            if not pending:
                return  # déjà parti avec un message précédent
            batch = sorted(pending, key=lambda p: p.message.id)
            pending.clear()
            last = batch[-1].message
            if len(batch) > 1:
                logger.info("%s messages regroupés en une seule réponse dans #%s", len(batch), channel)
            # Le statut en cours est aussi gardé en mémoire, comme le reste du message.
            parts = [(p.user_text, p.image) for p in batch]
            lines = [p.line for p in batch]
            notes = [self.presence.late_note(late, several=len(batch) > 1)] if late else []
            if note := self.presence_note():
                notes.append(note)
            for note in reversed(notes):
                parts.insert(0, (note, None))
                lines.insert(0, {"content": note})
            try:
                async with typing(channel):
                    try:
                        answer = await self.claude.ask(self.history.get(channel.id), parts,
                                                       emojis=emoji_names(last.guild))
                    except ClaudeError as exc:
                        await self.safe_reply(last, exc.user_message)
                        return
                    except Exception:
                        logger.exception("Erreur inattendue pendant le traitement d'un message")
                        await self.safe_reply(last, "Désolé, une erreur inattendue est survenue.")
                        return
            finally:
                self.forget_waiting(*(p.message for p in batch))
            await self.history.add_exchange(channel.id, lines, short_emojis(answer.text), answer.web_used)
        await self.send_answer(last, answer.text)

    async def handle_mention(self, message: discord.Message) -> None:
        if not self.available() or message.channel.id in self._waiters:
            self._mentions.setdefault(message.channel.id, []).append(message)
            self.answer_later(message, self.answer_mentions)
            return
        await self.answer_mention(message)

    async def answer_mentions(self, channel: discord.abc.Messageable, late: Status | None = None) -> None:
        """Répond, une par une, aux mentions reçues pendant une absence (avec « Répondre »)."""
        for message in sorted(self._mentions.pop(channel.id, []), key=lambda m: m.id):
            try:
                await self.answer_mention(message, late)
            finally:
                self.forget_waiting(message)

    async def answer_mention(self, message: discord.Message, late: Status | None = None) -> None:
        """Hors du salon dédié : répond à une mention en lisant les messages récents."""
        lines: list[str] = []
        if self.settings.context_message_limit > 0:
            try:
                async for msg in message.channel.history(limit=self.settings.context_message_limit, before=message):
                    content = short_emojis(msg.clean_content.strip())
                    if msg.attachments:
                        content = (content + " [pièce jointe]").strip()
                    if msg.stickers:
                        content = (content + " [sticker]").strip()
                    if content:
                        lines.append(f"{msg.author.display_name} : {content}")
            except discord.Forbidden:
                logger.warning("Pas la permission de lire l'historique de #%s.", message.channel)
        lines.reverse()

        # Seuls la première image (à défaut, le premier PDF) et le premier fichier texte de la mention
        # sont lus (les autres pièces jointes sont ignorées). L'image du message cité ne sert que s'il
        # n'y a ni image ni PDF.
        images = [a for a in message.attachments if looks_like_image(a)]
        pdfs = [a for a in message.attachments if looks_like_pdf(a)] if self.settings.pdf_max_pages else []
        try:
            if images:
                image = await self.download_image(images[0])
            else:
                image = await self.download_pdf(pdfs[0]) if pdfs else None
        except (ImageError, PdfError) as exc:
            await self.safe_reply(message, exc.user_message)
            return
        reply, reply_image = await self.read_reply(message, want_image=image is None)
        image = image or reply_image

        bot_name = f"@{message.guild.me.display_name}" if message.guild else f"@{self.user.name}"
        text = short_emojis(message.clean_content.replace(bot_name, "").strip()) or "(simple mention, sans texte)"
        channel_name = f"#{message.channel.name}" if hasattr(message.channel, "name") else "un message privé"

        parts = []
        if late:
            parts.append(self.presence.late_note(late, several=False))
        if note := self.presence_note():
            parts.append(note)
        if lines:
            parts.append(f"Contexte : derniers messages dans {channel_name}\n" + "\n".join(lines))
        reply_part = f" ({reply})" if reply else ""
        parts.append(f"{self.timestamp(message)}, {message.author.display_name} te mentionne{reply_part} et écrit : {text}")
        prompt = "\n\n".join(parts)

        text_files = [a for a in message.attachments if looks_like_text(a)]
        if text_files:
            try:
                prompt += text_file_prompt(*await self.download_text_file(text_files[0]))
            except TextFileError as exc:
                await self.safe_reply(message, exc.user_message)
                return

        async with typing(message.channel):
            try:
                answer = await self.claude.ask([], [(prompt, image)], emojis=emoji_names(message.guild),
                                               cache_history=False)
            except ClaudeError as exc:
                await self.safe_reply(message, exc.user_message)
                return
            except Exception:
                logger.exception("Erreur inattendue pendant le traitement d'une mention")
                await self.safe_reply(message, "Désolé, une erreur inattendue est survenue.")
                return
        # En retard, la réponse cite la mention : on sait à quoi elle répond.
        await self.send_answer(message, answer.text, reply=late is not None)

    # ------------------------------------------------------ réponses différées

    def available(self) -> bool:
        return self.presence is None or self.presence.available

    def answer_later(self, message: discord.Message, answer) -> None:
        """Met le message de côté : answer(salon, late=statut) y répondra quand le bot le pourra."""
        self.remember_waiting(message)
        channel = message.channel
        if channel.id in self._waiters:
            return  # une attente est déjà en cours dans ce salon : le message en profitera
        status = self.presence.status
        logger.info("Réponse différée dans #%s : %s.", channel, status.unavailability())
        self._waiters[channel.id] = asyncio.create_task(self._answer_when_possible(channel, answer, status))

    async def _answer_when_possible(self, channel: discord.abc.Messageable, answer, status: Status) -> None:
        try:
            await self.presence.wait_turn()
        finally:
            self._waiters.pop(channel.id, None)
        try:
            await answer(channel, late=status)
        except Exception:
            logger.exception("Erreur inattendue pendant une réponse différée")

    async def on_status_change(self, before: Status, after: Status) -> None:
        """Passe absent ou en « Ne pas déranger » : prévient le salon dédié si une conversation y est en cours."""
        if not before.available or after.available or self._last_main_message is None:
            return
        if discord.utils.utcnow() - self._last_main_message > DEPARTURE_WINDOW:
            return
        channel = self.get_channel(self.settings.channel_id)
        if channel is None:
            return
        async with self.history.lock(channel.id):
            # Attend la fin d'une réponse en cours ; le statut a pu changer encore entre-temps.
            if self.presence.status is not after:
                return
            parts = [(note, None) for note in (self.presence_note(), self.presence.departure_note()) if note]
            try:
                async with typing(channel):
                    answer = await self.claude.ask(self.history.get(channel.id), parts,
                                                   emojis=emoji_names(channel.guild))
            except ClaudeError as exc:
                logger.warning("Message d'absence non envoyé : %s", exc.user_message)
                return
            lines = [{"content": text} for text, _ in parts]
            await self.history.add_exchange(channel.id, lines, short_emojis(answer.text), answer.web_used)
        logger.info("Conversation en cours dans #%s : le salon est prévenu de l'absence.", channel)
        for chunk in split_message(discord_emojis(short_emojis(answer.text), channel.guild)):
            try:
                await channel.send(chunk)
            except discord.HTTPException as exc:
                logger.error("Envoi du message d'absence impossible : %s", exc)
                return

    def remember_waiting(self, message: discord.Message) -> None:
        self._waiting.add((message.channel.id, message.id))
        self._save_waiting()

    def forget_waiting(self, *messages: discord.Message) -> None:
        keys = {(m.channel.id, m.id) for m in messages}
        if keys & self._waiting:
            self._waiting -= keys
            self._save_waiting()

    def _save_waiting(self) -> None:
        try:
            self._waiting_file.write_text(json.dumps(sorted(self._waiting)), encoding="utf-8")
        except OSError as exc:
            logger.warning("Impossible d'écrire %s : %s", self._waiting_file, exc)

    async def replay_waiting(self) -> None:
        """Au démarrage : reprend les messages restés sans réponse avant le redémarrage."""
        try:
            saved = json.loads(self._waiting_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning("%s illisible, ignoré : %s", self._waiting_file, exc)
            return
        messages = []
        for channel_id, message_id in saved:
            try:
                channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
                messages.append(await channel.fetch_message(message_id))
            except discord.HTTPException:
                pass  # message supprimé, salon inaccessible…
        self._waiting.clear()
        self._save_waiting()
        if messages:
            logger.info("%d message(s) resté(s) sans réponse avant le redémarrage : reprise.", len(messages))
            # En parallèle, comme à leur arrivée : ceux du salon dédié sont regroupés.
            await asyncio.gather(*(self.on_message(m) for m in sorted(messages, key=lambda m: m.id)),
                                 return_exceptions=True)

    # ------------------------------------------------------------- réactions

    async def reply_to_sticker(self, message: discord.Message) -> bool:
        """Répond à un sticker par un sticker aléatoire du serveur (False si aucun n'est utilisable)."""
        if not message.stickers or message.guild is None:
            return False
        stickers = [s for s in message.guild.stickers if s.available]
        random.shuffle(stickers)
        for sticker in stickers:
            try:
                await message.channel.send(stickers=[sticker])
                logger.info("Sticker envoyé : %s", sticker.name)
                return True
            except discord.Forbidden as exc:  # pas le droit d'envoyer de sticker : inutile d'insister
                logger.warning("Envoi de sticker interdit : %s", exc)
                break
            except discord.HTTPException as exc:
                logger.warning("Sticker %s inutilisable : %s", sticker.name, exc)
        return False

    async def reply_to_emoji(self, message: discord.Message) -> bool:
        """Répond à un message composé d'un seul emoji personnalisé par un emoji aléatoire du serveur
        (False si le serveur n'en a aucun)."""
        if not CUSTOM_EMOJI_ONLY.match(message.content.strip()):
            return False
        emojis = usable_emojis(message.guild)
        if not emojis:
            return False
        try:
            await message.channel.send(str(random.choice(emojis)))
        except discord.HTTPException as exc:
            logger.warning("Impossible d'envoyer l'emoji : %s", exc)
        return True

    # ------------------------------------------------------- pièces jointes

    async def read_attachments(
        self, message: discord.Message
    ) -> tuple[ImageData | None, tuple[str, str] | None] | bool:
        """Vérifie les pièces jointes, puis télécharge l'image (ou le PDF) et le fichier texte.

        Renvoie (image ou PDF, (nom, contenu) du fichier texte), chacun pouvant valoir None,
        ou False si l'envoi est refusé (l'utilisateur est alors prévenu).
        """
        if not message.attachments:
            return None, None

        images = [a for a in message.attachments if looks_like_image(a)]
        pdfs = [a for a in message.attachments if looks_like_pdf(a)] if self.settings.pdf_max_pages else []
        texts = [a for a in message.attachments if looks_like_text(a)]
        others = [a.filename for a in message.attachments if a not in images + pdfs + texts]
        if others:
            kinds = "des images, des PDF" if self.settings.pdf_max_pages else "des images"
            await self.safe_reply(message, f"Je ne peux lire que {kinds} et des fichiers texte (.txt, .md, .csv, code…). "
                                           f"Fichier(s) non pris en charge : {', '.join(others)}.")
            return False
        if len(images) > 1:
            await self.safe_reply(message, "Une seule image par message, s'il te plaît.")
            return False
        if pdfs and (images or len(pdfs) > 1):
            await self.safe_reply(message, "Un seul PDF par message, sans image, s'il te plaît.")
            return False
        if len(texts) > 1:
            await self.safe_reply(message, "Un seul fichier texte par message, s'il te plaît.")
            return False

        try:
            if images:
                image = await self.download_image(images[0])
            else:
                image = await self.download_pdf(pdfs[0]) if pdfs else None
            text_file = await self.download_text_file(texts[0]) if texts else None
        except (ImageError, PdfError, TextFileError) as exc:
            await self.safe_reply(message, exc.user_message)
            return False
        return image, text_file

    async def download_text_file(self, attachment: discord.Attachment) -> tuple[str, str]:
        """Télécharge un fichier texte et renvoie (nom, contenu) (lève TextFileError en cas de problème)."""
        if attachment.size > TEXT_FILE_MAX_BYTES:
            raise TextFileError(f"Ce fichier texte est trop long (maximum {TEXT_FILE_MAX_BYTES // 1000} Ko).")

        try:
            data = await attachment.read()
        except discord.HTTPException as exc:
            logger.error("Téléchargement du fichier texte impossible : %s", exc)
            raise TextFileError("Je n'ai pas réussi à télécharger ton fichier, réessaie.")

        return attachment.filename, decode_text_file(data)

    async def download_pdf(self, attachment: discord.Attachment) -> ImageData:
        """Télécharge et vérifie un PDF (lève PdfError en cas de problème)."""
        if attachment.size > PDF_MAX_BYTES:
            raise PdfError(f"Ce PDF est trop lourd (maximum {PDF_MAX_BYTES // 1_000_000} Mo).")

        try:
            data = await attachment.read()
        except discord.HTTPException as exc:
            logger.error("Téléchargement du PDF impossible : %s", exc)
            raise PdfError("Je n'ai pas réussi à télécharger ton PDF, réessaie.")

        # Lecture du PDF dans un thread : ne bloque pas le bot.
        return await asyncio.to_thread(prepare_pdf, attachment.filename, data, self.settings.pdf_max_pages)

    async def download_image(self, attachment: discord.Attachment) -> ImageData:
        """Télécharge et prépare une image (lève ImageError en cas de problème)."""
        if attachment.size > self.settings.max_image_size:
            max_mb = self.settings.max_image_size / 1_000_000
            raise ImageError(f"Cette image est trop lourde (maximum {max_mb:.2f} Mo).")

        try:
            data = await attachment.read()
        except discord.HTTPException as exc:
            logger.error("Téléchargement de l'image impossible : %s", exc)
            raise ImageError("Je n'ai pas réussi à télécharger ton image, réessaie.")

        # Réduit / convertit l'image si besoin (dans un thread : ne bloque pas le bot).
        return await asyncio.to_thread(prepare_image, data)

    # ------------------------------------------- message cité ou transféré

    async def fetch_cited(self, message: discord.Message) -> discord.Message | None:
        """Renvoie le message auquel ce message répond (bouton « Répondre » de Discord),
        ou None s'il ne répond à rien ou si le message cité est inaccessible."""
        ref = message.reference
        if ref is None or ref.message_id is None or ref.type != discord.MessageReferenceType.reply:
            return None

        cited = ref.resolved
        if cited is None:  # Discord ne l'a pas fourni (vieux message) : on le télécharge
            try:
                cited = await message.channel.fetch_message(ref.message_id)
            except discord.HTTPException as exc:
                logger.info("Message cité inaccessible : %s", exc)
                return None
            ref.resolved = cited  # évite de le télécharger une seconde fois
        return cited if isinstance(cited, discord.Message) else None  # None : message cité supprimé

    async def replies_to_me(self, message: discord.Message) -> bool:
        cited = await self.fetch_cited(message)
        return cited is not None and cited.author == self.user

    async def read_reply(self, message: discord.Message, want_image: bool) -> tuple[str | None, ImageData | None]:
        """Lit le message auquel l'utilisateur répond.

        Renvoie sa description pour Claude, ainsi que son image si want_image est vrai,
        ou (None, None) si le message ne répond à rien ou si le message cité est inaccessible.
        """
        cited = await self.fetch_cited(message)
        if cited is None:
            return None, None

        image = await self.first_image(cited.attachments, "cité") if want_image else None
        description = self.describe_quoted(cited.clean_content, image, cited.attachments, cited.stickers)

        who = "à ton message" if cited.author == self.user else f"au message de {cited.author.display_name}"
        return f"en réponse {who} du {self.timestamp(cited)} : {description or '(message vide)'}", image

    async def read_forward(self, message: discord.Message) -> tuple[str | None, ImageData | None]:
        """Lit le message transféré (bouton « Transférer » de Discord).

        Renvoie sa description pour Claude et son image, ou (None, None) si le message
        n'est pas un transfert ou si le message transféré ne contient rien de lisible.
        """
        if not message.message_snapshots:
            return None, None
        snapshot = message.message_snapshots[0]

        image = await self.first_image(snapshot.attachments, "transféré")
        # Contrairement à un message normal, le contenu transféré garde ses mentions brutes (<@123>).
        content = self.clean_mentions(snapshot.content)
        description = self.describe_quoted(content, image, snapshot.attachments, snapshot.stickers)
        if not description:
            return None, None
        return f"du {self.timestamp(snapshot)} : {description}", image

    async def first_image(self, attachments: list[discord.Attachment], kind: str) -> ImageData | None:
        """Télécharge la première image d'un message cité ou transféré (None si aucune ou illisible)."""
        images = [a for a in attachments if looks_like_image(a)]
        if not images:
            return None
        try:
            return await self.download_image(images[0])
        except ImageError as exc:
            logger.warning("Image du message %s ignorée : %s", kind, exc.user_message)
            return None

    @staticmethod
    def describe_quoted(content: str, image: ImageData | None, attachments: list, stickers: list) -> str:
        """Description d'un message cité ou transféré (vide s'il n'y a rien de lisible)."""
        content = short_emojis(content.strip())
        if len(content) > QUOTE_MAX_LENGTH:
            content = content[:QUOTE_MAX_LENGTH].rstrip() + "…"
        parts = [f"« {content} »"] if content else []
        if image is not None:
            parts.append("[son image est jointe à ce message]")
        elif attachments:
            parts.append("[pièce jointe]")
        if stickers:
            parts.append("[sticker]")
        return " ".join(parts)

    def clean_mentions(self, text: str) -> str:
        """Remplace les mentions brutes (<@123>) par des pseudos, comme le fait clean_content."""
        def name(match: re.Match) -> str:
            user = self.get_user(int(match.group(1)))
            return f"@{user.display_name}" if user else "@quelqu'un"
        return USER_MENTION.sub(name, text)

    # ------------------------------------------------------------- envoi

    def presence_note(self) -> str | None:
        return self.presence.prompt_note() if self.presence else None

    @staticmethod
    def message_body(message: discord.Message) -> str:
        """Texte d'un message du salon dédié, tel qu'il est conservé en mémoire (sans ce qui le précède)."""
        text = short_emojis(message.clean_content.strip())
        if message.stickers:
            text = f"{text} [sticker « {message.stickers[0].name} »]".strip()
        return text or "(message sans texte)"

    def timestamp(self, message: discord.Message | discord.MessageSnapshot) -> str:
        return message.created_at.astimezone(self.settings.timezone).strftime("%d/%m/%Y %H:%M")

    async def send_answer(self, message: discord.Message, answer: str, reply: bool = False) -> None:
        # Comme un humain : la réponse est postée à la suite dans le salon, sans « Répondre »
        # (sauf reply=True : le premier morceau répond alors au message).
        # Les emojis passent par leur nom : un code écrit avec un mauvais identifiant est ainsi corrigé.
        answer = discord_emojis(short_emojis(answer), message.guild)
        for i, chunk in enumerate(split_message(answer)):
            # fail_if_not_exists=False : si le message a été supprimé entre-temps, envoi normal.
            reference = message.to_reference(fail_if_not_exists=False) if reply and i == 0 else None
            try:
                await message.channel.send(chunk, reference=reference, mention_author=False)
            except discord.HTTPException as exc:
                logger.error("Envoi de la réponse impossible : %s", exc)
                return

    async def safe_reply(self, message: discord.Message, text: str) -> None:
        try:
            await message.reply(text, mention_author=False)
        except discord.HTTPException:
            try:
                await message.channel.send(text)
            except discord.HTTPException as exc:
                logger.error("Impossible d'envoyer un message : %s", exc)


# ------------------------------------------------------------- commandes slash

@app_commands.command(name="reset", description="Réinitialise la mémoire du bot dans ce salon")
@app_commands.guild_only()
# Masque la commande aux membres sans « Gérer les messages » (modérateurs) ;
# les administrateurs ont implicitement toutes les permissions.
@app_commands.default_permissions(manage_messages=True)
async def reset_command(interaction: discord.Interaction) -> None:
    # Vérification côté bot, au cas où un admin aurait ouvert la commande à tous dans les paramètres du serveur.
    perms = interaction.permissions
    if not (perms.manage_messages or perms.administrator):
        logger.info("Tentative de /reset refusée pour %s dans le salon %s", interaction.user, interaction.channel_id)
        await interaction.response.send_message(
            "Seuls les modérateurs et administrateurs du serveur peuvent utiliser cette commande.",
            ephemeral=True,
        )
        return
    bot: ChatBot = interaction.client  # type: ignore[assignment]
    # Le reset attend la fin d'une éventuelle réponse en cours, ce qui peut dépasser
    # les 3 secondes accordées par Discord pour répondre à une commande.
    await interaction.response.defer()
    await bot.history.reset(interaction.channel_id)
    logger.info("Historique réinitialisé dans le salon %s par %s", interaction.channel_id, interaction.user)
    await interaction.followup.send("L'historique de conversation a été réinitialisé.")
