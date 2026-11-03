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
from .config import Settings, memory_warning
from .emojis import discord_emojis, emoji_names, short_emojis, usable_emojis
from .history import ConversationHistory, images_marker, render_line
from .images import ImageError, prepare_image
from .members import MemberRegistry, with_username
from .memory import DATE_FORMAT, MemoryStore
from .memory_views import MemoryCard, member_target, reply_error, see_user, server_target
from .models import display_name, is_known
from .pdf_files import PDF_EXTENSIONS, PDF_MAX_BYTES, PdfError, prepare_pdf
from .presence import PresenceManager, Status
from .text_files import TEXT_EXTENSIONS, TEXT_FILE_MAX_BYTES, TextFileError, decode_text_file
from .text_utils import split_message

logger = logging.getLogger("chatbot.discord")

CUSTOM_EMOJI_ONLY = re.compile(r"^<a?:\w+:\d+>$")
USER_MENTION = re.compile(r"<@!?(\d+)>")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff")
# Couleur du message de démarrage : le vert du statut « en ligne » de Discord.
ONLINE_COLOR = discord.Color(0x23A55A)
# Longueur maximale du texte d'un message cité ou transféré transmis à Claude.
QUOTE_MAX_LENGTH = 500
# Images lues par message : chacune coûte jusqu'à ~1 600 tokens.
MAX_IMAGES = 4
# Messages écrits par quelqu'un ; les autres types sont des messages système
# (arrivée d'un membre, message épinglé, boost, fil créé…).
USER_MESSAGE_TYPES = (discord.MessageType.default, discord.MessageType.reply)
# En passant absent ou en « Ne pas déranger », le bot prévient son salon si on y a parlé depuis moins de :
DEPARTURE_WINDOW = timedelta(minutes=3)


def author_name(user: discord.User | discord.Member) -> str:
    """Nom d'un auteur pour Claude, avec son nom d'utilisateur (voir with_username)."""
    return with_username(user.display_name, user.name)


def looks_like_image(attachment: discord.Attachment) -> bool:
    return (attachment.content_type or "").startswith("image/") or attachment.filename.lower().endswith(IMAGE_EXTENSIONS)


def looks_like_text(attachment: discord.Attachment) -> bool:
    return attachment.filename.lower().endswith(TEXT_EXTENSIONS)


def looks_like_pdf(attachment: discord.Attachment) -> bool:
    return attachment.content_type == "application/pdf" or attachment.filename.lower().endswith(PDF_EXTENSIONS)


def channel_problem(client: discord.Client, channel_id: int) -> str | None:
    """Pourquoi le salon dédié (CHANNEL_ID) est inutilisable, ou None s'il convient."""
    channel = client.get_channel(channel_id)
    if channel is None:
        if client.get_guild(channel_id):
            return (f"CHANNEL_ID ({channel_id}) est l'identifiant du serveur, pas du salon : fais un clic droit "
                    "sur le salon > Copier l'identifiant du salon.")
        return (f"Salon {channel_id} introuvable : vérifie CHANNEL_ID (clic droit sur le salon > Copier l'identifiant "
                "du salon, avec le mode développeur de Discord) et que le bot est bien invité sur ce serveur.")
    guild = getattr(channel, "guild", None)
    if guild is None or not isinstance(channel, discord.abc.Messageable):
        kind = ("une catégorie" if isinstance(channel, discord.CategoryChannel)
                else "un forum" if isinstance(channel, discord.ForumChannel)
                else "une conversation privée" if guild is None else "un salon où l'on ne peut pas écrire")
        return f"CHANNEL_ID ({channel_id}) désigne {kind} : donne l'identifiant d'un salon textuel."
    permissions = channel.permissions_for(guild.me)
    in_thread = isinstance(channel, discord.Thread)
    needed = (("view_channel", "Voir les salons"),
              ("send_messages_in_threads" if in_thread else "send_messages",
               "Envoyer des messages dans les fils" if in_thread else "Envoyer des messages"),
              ("read_message_history", "Voir les anciens messages"))
    if missing := [label for name, label in needed if not getattr(permissions, name)]:
        names = ", ".join(f"« {label} »" for label in missing)
        plural = "les permissions" if len(missing) > 1 else "la permission"
        return (f"Il manque au bot {plural} {names} dans #{channel} : ajoute-{'les' if len(missing) > 1 else 'la'} "
                "à son rôle ou dans les permissions du salon.")
    return None


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


def attachments_marker(attachments: list[ImageData]) -> str:
    """Remplace les images ou le PDF d'un message dans la mémoire (leur contenu n'est pas conservé)."""
    if attachments[0].is_pdf:
        return f"[PDF joint « {attachments[0].name} » : tu l'as lu à ce moment-là, mais son contenu n'est plus conservé en mémoire]"
    return images_marker(len(attachments))


@dataclass
class PendingMessage:
    """Message du salon dédié, prêt à être envoyé à Claude."""
    message: discord.Message
    line: dict   # ligne conservée en mémoire (voir history.render_line)
    extra: str   # ajouté pour Claude seulement : contenu du fichier texte joint
    images: list[ImageData]  # images, ou un seul PDF

    @property
    def user_text(self) -> str:
        """Texte envoyé à Claude : les images ou le fichier eux-mêmes remplacent leur repère."""
        line = {key: value for key, value in self.line.items() if key != "attachments"}
        return render_line(line) + self.extra


class HomeTree(app_commands.CommandTree):
    """Commandes slash acceptées seulement sur le serveur du bot (voir ChatBot.home_guild_id) : un
    administrateur d'un autre serveur où le bot a été invité ne peut ni lire ni modifier la mémoire."""

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        bot: ChatBot = self.client  # type: ignore[assignment]
        if interaction.guild_id is not None and interaction.guild_id == bot.home_guild_id():
            return True
        logger.warning("/%s refusée pour %s : serveur %s, hors du serveur du bot",
                       (interaction.data or {}).get("name", "?"), interaction.user, interaction.guild_id)
        if interaction.type is discord.InteractionType.application_command:
            await interaction.response.send_message("Ce bot ne fonctionne que sur son propre serveur.",
                                                    ephemeral=True)
        return False

    async def on_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        logger.error("Erreur pendant /%s", (interaction.data or {}).get("name", "?"), exc_info=error)
        await reply_error(interaction)


class ChatBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True  # nécessaire pour lire le texte des messages
        super().__init__(
            command_prefix=commands.when_mentioned,  # aucune commande textuelle utilisée
            intents=intents,
            help_command=None,
            tree_cls=HomeTree,
            # Le bot ne peut jamais pinger @everyone, @here, des rôles ou des membres.
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.members = MemberRegistry(settings.memory_dir / "registre.json") if settings.memory else None
        if self.members and self.members.broken:
            self.members = None  # registre illisible : mémoire coupée comme avec MEMORY=off (voir MemberRegistry)
        self.memory = MemoryStore(settings.memory_dir, self.members, settings.memory_member_limit,
                                  settings.memory_server_limit, settings.timezone) if self.members else None
        self.claude = ClaudeClient(settings, self.memory)
        self.history = ConversationHistory(settings.history_file, settings.max_history_length)
        self._pending: dict[int, list[PendingMessage]] = {}  # messages en attente de réponse, par salon
        self._announced = False
        self.config_error: str | None = None  # salon dédié inutilisable : main.py sort en erreur
        self._memory_warning = memory_warning(settings) if self.memory else None
        self.presence =PresenceManager(self, self.claude, settings) if settings.presence else None
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
        if self.presence:
            self.tree.add_command(skip_command)
        # Toujours présentes : avec MEMORY=off, elles indiquent que la mémoire est coupée.
        self.tree.add_command(memory_command)
        self.tree.add_command(server_memory_command)
        await self.sync_commands()
        if self.memory:
            # Lecture de toutes les fiches, avant la connexion : elle ne bloque aucun message.
            await asyncio.to_thread(self.memory.index)
        s = self.settings
        memory = "off" if not s.memory else f"on ({s.memory_dir})" if self.memory else "coupée (registre illisible)"
        logger.info("Modèle %s — réflexion : %s, effort : %s, recherche web : %s, mémoire des membres : %s",
                    s.model, "on" if s.thinking else "off", s.effort or "par défaut",
                    "on" if s.web_search else "off", memory)
        if not is_known(s.model):
            # Un nom mal écrit ne serait sinon découvert qu'au premier message (erreur 404).
            prices = ""
            if s.price_input_per_mtok is None or s.price_output_per_mtok is None:
                prices = (" S'il existe bien, renseigne PRICE_INPUT_PER_MTOK et PRICE_OUTPUT_PER_MTOK "
                          "pour estimer les coûts.")
            logger.warning("Modèle %s inconnu du bot : vérifie CLAUDE_MODEL dans le .env.%s", s.model, prices)
        # Une réponse qui remplit MAX_TOKENS s'écrit à environ 80 tokens par seconde au mieux ;
        # les recherches web et les pages lues ajoutent leur propre attente.
        needed = s.max_tokens // 80 + (40 if s.web_search else 0)
        if s.api_timeout < needed:
            logger.warning("API_TIMEOUT (%g s) est court pour MAX_TOKENS=%s%s : une longue réponse serait "
                           "abandonnée (« Je mets trop de temps à répondre »). Mets au moins %s dans le .env.",
                           s.api_timeout, s.max_tokens, " avec la recherche web" if s.web_search else "", needed)
        if self._memory_warning:
            logger.warning("%s", self._memory_warning)  # rappelé dans le message « En ligne »
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
        if not self._announced:  # simple reconnexion : on ne reposte pas le message de démarrage
            await self.announce()

    async def on_guild_available(self, guild: discord.Guild) -> None:
        # Serveur du salon indisponible au démarrage (panne Discord), revenu depuis : le bot s'annonce enfin.
        if self.is_ready() and not self._announced and self.home_guild_id() == guild.id:
            await self.announce()

    async def announce(self) -> None:
        """Vérifie le salon dédié, puis y poste le message de démarrage. Un salon inutilisable arrête le bot."""
        if problem := channel_problem(self, self.settings.channel_id):
            if self.get_channel(self.settings.channel_id) is None and any(g.unavailable for g in self.guilds):
                # Le salon est peut-être sur ce serveur : Discord n'en envoie le contenu qu'à son retour.
                logger.warning("Salon %s introuvable pour l'instant : un serveur du bot est indisponible chez "
                               "Discord. Le bot s'annoncera dès son retour.", self.settings.channel_id)
                return
            logger.critical("Erreur de configuration : %s", problem)
            self.config_error = problem
            await self.close()
            return
        self._announced = True
        self._replay_task = asyncio.create_task(self.replay_waiting())  # référence gardée : pas de ramasse-miettes

        channel = self.get_channel(self.settings.channel_id)
        if others := [guild for guild in self.guilds if guild.id != self.home_guild_id()]:
            logger.warning("Le bot est aussi sur %d autre(s) serveur(s), où il ne répond pas : %s. Retire-le de ces "
                           "serveurs, et décoche Public Bot dans le portail Discord (voir le README).", len(others),
                           ", ".join(f"{guild.name} ({guild.id})" for guild in others))
        me = channel.guild.me if getattr(channel, "guild", None) else self.user
        logger.info("%d emoji(s) du serveur utilisable(s) dans les réponses.",
                    len(usable_emojis(getattr(channel, "guild", None))))
        started = discord.utils.utcnow()
        embed = discord.Embed(
            title="En ligne",
            description=f"**{me.display_name}** vient de démarrer et répond de nouveau aux messages.",
            color=ONLINE_COLOR,
            timestamp=started,
        )
        embed.set_thumbnail(url=me.display_avatar.url)
        embed.add_field(name="Version", value=f"`{VERSION}`")
        embed.add_field(name="Modèle", value=display_name(self.settings.model))
        embed.add_field(name="Démarrage", value=discord.utils.format_dt(started, "R"))
        if self._memory_warning:
            # Sur OpenShift, personne ne lit le journal de démarrage.
            embed.add_field(name="⚠️ Réglage à revoir", value=self._memory_warning, inline=False)
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

    def home_guild_id(self) -> int | None:
        """Le serveur du salon dédié (CHANNEL_ID), le seul où le bot répond : la mémoire n'existe qu'en
        un exemplaire. None tant que le salon est introuvable (avant la connexion, CHANNEL_ID faux)."""
        guild = getattr(self.get_channel(self.settings.channel_id), "guild", None)
        return guild.id if guild else None

    async def on_guild_join(self, guild: discord.Guild) -> None:
        logger.warning("Bot ajouté au serveur %s (%s), où il ne répond pas : retire-le de ce serveur, et "
                       "décoche Public Bot dans le portail Discord (voir le README).", guild.name, guild.id)

    # ------------------------------------------------------------- messages

    async def on_message(self, message: discord.Message) -> None:
        # Ignore ses propres messages et les webhooks : évite les boucles infinies.
        if message.author == self.user or message.webhook_id is not None:
            return
        # Ni autre serveur ni message privé : réponses payées par le propriétaire, mémoire du serveur.
        if message.guild is None or message.guild.id != self.home_guild_id():
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
        if (in_main_channel or mentioned) and self.members:
            await self.register_members(message)

        if in_main_channel:
            await self.handle_main_channel(message)
        elif mentioned:
            await self.handle_mention(message)

    async def concerned_members(self, message: discord.Message) -> list[discord.User | discord.Member]:
        """Membres concernés par un message, sans les bots : auteur, auteur du message cité, membres
        mentionnés (ordre des fiches jointes quand elles sont trop nombreuses)."""
        users = [message.author]
        if (cited := await self.fetch_cited(message)) is not None:
            users.append(cited.author)
        users += message.mentions
        unique = {user.id: user for user in users if not user.bot}
        return list(unique.values())

    async def member_notes(self, messages: list[discord.Message]) -> str:
        """Fiches des membres concernés par ces messages, à joindre pour Claude (vide sans mémoire)."""
        if not self.memory:
            return ""
        ids = [user.id for message in messages for user in await self.concerned_members(message)]
        return self.memory.notes_for(ids, author_ids=[message.author.id for message in messages])

    async def register_members(self, message: discord.Message) -> None:
        """Ajoute au registre les membres concernés par le message, ou met leurs noms à jour."""
        for user in await self.concerned_members(message):
            await see_user(self.members, user)

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

        forward, forward_images = await self.read_forward(message)
        if not (content or message.attachments or message.stickers or forward):
            return  # sondage, transfert d'un message vide… : rien à lire

        attachments = await self.read_attachments(message)
        if attachments is False:
            return  # pièce jointe refusée, l'utilisateur a été prévenu
        images, text_file = attachments

        # Les images du message cité ne sont utilisées que si le message n'en contient pas lui-même.
        reply, reply_images = await self.read_reply(message, want_images=not (images or forward_images))
        images = images or forward_images or reply_images

        line: dict = {"id": message.id, "date": self.timestamp(message), "author": author_name(message.author)}
        if forward:
            line["forward"] = forward
        else:
            if reply:
                line["reply_to"] = message.reference.message_id  # citation retirée si ce message est supprimé
                line["reply"] = reply
            line["content"] = self.message_body(message)
        markers = [attachments_marker(images)] if images else []
        extra = ""
        if text_file:
            extra = text_file_prompt(*text_file)
            markers.append(text_file_marker(text_file[0]))
        if markers:
            line["attachments"] = markers

        # Un seul appel à la fois par salon : l'historique reste cohérent. Les messages arrivés
        # pendant une réponse attendent dans la file, puis partent ensemble dans un seul appel.
        self._pending.setdefault(message.channel.id, []).append(PendingMessage(message, line, extra, images))
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
            parts = [(p.user_text, p.images) for p in batch]
            lines = [p.line for p in batch]
            notes = [self.presence.late_note(late, several=len(batch) > 1)] if late else []
            if note := self.presence_note():
                notes.append(note)
            for note in reversed(notes):
                parts.insert(0, (note, []))
                lines.insert(0, {"content": note})
            try:
                async with typing(channel):
                    try:
                        notes = await self.member_notes([p.message for p in batch])
                        answer = await self.claude.ask(self.history.get(channel.id), parts,
                                                       emojis=emoji_names(last.guild), member_notes=notes)
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
                        lines.append(f"{author_name(msg.author)} : {content}")
            except discord.Forbidden:
                logger.warning("Pas la permission de lire l'historique de #%s.", message.channel)
        lines.reverse()

        # Seuls les MAX_IMAGES premières images (à défaut, le premier PDF) et le premier fichier texte
        # de la mention sont lus (les autres pièces jointes sont ignorées). Les images du message cité
        # ne servent que s'il n'y a ni image ni PDF.
        image_files = [a for a in message.attachments if looks_like_image(a)][:MAX_IMAGES]
        pdfs = [a for a in message.attachments if looks_like_pdf(a)] if self.settings.pdf_max_pages else []
        try:
            if image_files:
                images = await self.download_images(image_files)
            else:
                images = [await self.download_pdf(pdfs[0])] if pdfs else []
        except (ImageError, PdfError) as exc:
            await self.safe_reply(message, exc.user_message)
            return
        reply, reply_images = await self.read_reply(message, want_images=not images)
        images = images or reply_images

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
        parts.append(f"{self.timestamp(message)}, {author_name(message.author)} te mentionne{reply_part} et écrit : {text}")
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
                answer = await self.claude.ask([], [(prompt, images)], emojis=emoji_names(message.guild),
                                               cache_history=False, member_notes=await self.member_notes([message]))
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
            parts = [(note, []) for note in (self.presence_note(), self.presence.departure_note()) if note]
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
    ) -> tuple[list[ImageData], tuple[str, str] | None] | bool:
        """Vérifie les pièces jointes, puis télécharge les images (ou le PDF) et le fichier texte.

        Renvoie (images ou PDF, éventuellement aucun ; (nom, contenu) du fichier texte ou None),
        ou False si l'envoi est refusé (l'utilisateur est alors prévenu).
        """
        if not message.attachments:
            return [], None

        images = [a for a in message.attachments if looks_like_image(a)]
        pdfs = [a for a in message.attachments if looks_like_pdf(a)] if self.settings.pdf_max_pages else []
        texts = [a for a in message.attachments if looks_like_text(a)]
        others = [a.filename for a in message.attachments if a not in images + pdfs + texts]
        if others:
            kinds = "des images, des PDF" if self.settings.pdf_max_pages else "des images"
            await self.safe_reply(message, f"Je ne peux lire que {kinds} et des fichiers texte (.txt, .md, .csv, code…). "
                                           f"Fichier(s) non pris en charge : {', '.join(others)}.")
            return False
        if len(images) > MAX_IMAGES:
            await self.safe_reply(message, f"{MAX_IMAGES} images par message au maximum, s'il te plaît.")
            return False
        if pdfs and (images or len(pdfs) > 1):
            await self.safe_reply(message, "Un seul PDF par message, sans image, s'il te plaît.")
            return False
        if len(texts) > 1:
            await self.safe_reply(message, "Un seul fichier texte par message, s'il te plaît.")
            return False

        try:
            if images:
                files = await self.download_images(images)
            else:
                files = [await self.download_pdf(pdfs[0])] if pdfs else []
            text_file = await self.download_text_file(texts[0]) if texts else None
        except (ImageError, PdfError, TextFileError) as exc:
            await self.safe_reply(message, exc.user_message)
            return False
        return files, text_file

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

    async def download_images(self, attachments: list[discord.Attachment]) -> list[ImageData]:
        """Télécharge et prépare plusieurs images en parallèle (lève ImageError si l'une pose problème).

        Si plusieurs échouent, l'erreur nomme la première dans l'ordre du message, pas la plus rapide.
        """
        results = await asyncio.gather(*(self.download_image(a) for a in attachments), return_exceptions=True)
        for attachment, result in zip(attachments, results):
            if isinstance(result, ImageError) and len(attachments) > 1:
                # précise laquelle, sinon « cette image » est ambigu
                raise ImageError(f"« {attachment.filename} » : {result.user_message}")
            if isinstance(result, BaseException):
                raise result
        return list(results)

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

    async def read_reply(self, message: discord.Message, want_images: bool) -> tuple[str | None, list[ImageData]]:
        """Lit le message auquel l'utilisateur répond.

        Renvoie sa description pour Claude, ainsi que ses images si want_images est vrai,
        ou (None, []) si le message ne répond à rien ou si le message cité est inaccessible.
        """
        cited = await self.fetch_cited(message)
        if cited is None:
            return None, []

        images = await self.quoted_images(cited.attachments, "cité") if want_images else []
        description = self.describe_quoted(cited.clean_content, images, cited.attachments, cited.stickers)

        who = "à ton message" if cited.author == self.user else f"au message de {author_name(cited.author)}"
        return f"en réponse {who} du {self.timestamp(cited)} : {description or '(message vide)'}", images

    async def read_forward(self, message: discord.Message) -> tuple[str | None, list[ImageData]]:
        """Lit le message transféré (bouton « Transférer » de Discord).

        Renvoie sa description pour Claude et ses images, ou (None, []) si le message
        n'est pas un transfert ou si le message transféré ne contient rien de lisible.
        """
        if not message.message_snapshots:
            return None, []
        snapshot = message.message_snapshots[0]

        images = await self.quoted_images(snapshot.attachments, "transféré")
        # Contrairement à un message normal, le contenu transféré garde ses mentions brutes (<@123>).
        content = self.clean_mentions(snapshot.content)
        description = self.describe_quoted(content, images, snapshot.attachments, snapshot.stickers)
        if not description:
            return None, []
        return f"du {self.timestamp(snapshot)} : {description}", images

    async def quoted_images(self, attachments: list[discord.Attachment], kind: str) -> list[ImageData]:
        """Télécharge les MAX_IMAGES premières images d'un message cité ou transféré (les illisibles sont ignorées)."""
        images = [a for a in attachments if looks_like_image(a)][:MAX_IMAGES]
        results = await asyncio.gather(*(self.download_image(a) for a in images), return_exceptions=True)
        for result in results:
            if isinstance(result, ImageError):
                logger.warning("Image du message %s ignorée : %s", kind, result.user_message)
            elif isinstance(result, BaseException):
                raise result
        return [r for r in results if isinstance(r, ImageData)]

    @staticmethod
    def describe_quoted(content: str, images: list[ImageData], attachments: list, stickers: list) -> str:
        """Description d'un message cité ou transféré (vide s'il n'y a rien de lisible)."""
        content = short_emojis(content.strip())
        if len(content) > QUOTE_MAX_LENGTH:
            content = content[:QUOTE_MAX_LENGTH].rstrip() + "…"
        parts = [f"« {content} »"] if content else []
        if len(images) > 1:
            parts.append(f"[ses {len(images)} images sont jointes à ce message]")
        elif images:
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
        return message.created_at.astimezone(self.settings.timezone).strftime(DATE_FORMAT)

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

async def allowed(interaction: discord.Interaction) -> bool:
    """Commande réservée aux administrateurs : prévient les autres et renvoie False."""
    # Vérification côté bot, au cas où un admin aurait ouvert la commande à tous dans les paramètres du serveur.
    if interaction.permissions.administrator:
        return True
    logger.info("Tentative de /%s refusée pour %s dans le salon %s",
                interaction.command.name if interaction.command else "?", interaction.user, interaction.channel_id)
    await interaction.response.send_message(
        "Seuls les administrateurs du serveur peuvent utiliser cette commande.",
        ephemeral=True,
    )
    return False


@app_commands.command(name="reset", description="Réinitialise la mémoire du bot dans ce salon")
@app_commands.guild_only()
# Masque la commande aux membres qui n'ont pas la permission Administrateur.
@app_commands.default_permissions(administrator=True)
async def reset_command(interaction: discord.Interaction) -> None:
    if not await allowed(interaction):
        return
    bot: ChatBot = interaction.client  # type: ignore[assignment]
    # Le reset attend la fin d'une éventuelle réponse en cours, ce qui peut dépasser
    # les 3 secondes accordées par Discord pour répondre à une commande.
    await interaction.response.defer()
    await bot.history.reset(interaction.channel_id)
    logger.info("Historique réinitialisé dans le salon %s par %s", interaction.channel_id, interaction.user)
    await interaction.followup.send("L'historique de conversation a été réinitialisé.")


@app_commands.command(name="skip", description="Passe tout de suite au statut Discord suivant du bot")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)  # comme /reset
async def skip_command(interaction: discord.Interaction) -> None:
    if not await allowed(interaction):
        return
    bot: ChatBot = interaction.client  # type: ignore[assignment]
    # Une pensée ou une raison d'absence est écrite par Claude : plus de 3 secondes possibles.
    await interaction.response.defer(ephemeral=True)
    try:
        status = await bot.presence.skip()
    except Exception:
        logger.exception("Erreur inattendue pendant /skip")
        await interaction.followup.send("Le statut n'a pas pu être changé, regarde les logs.", ephemeral=True)
        return
    if status is None:
        await interaction.followup.send("C'est la nuit : le bot dort, il n'y a pas de statut à passer.", ephemeral=True)
        return
    logger.info("Statut passé par %s", interaction.user)
    remaining = bot.presence.remaining()
    duration = f", pendant environ {remaining}" if remaining else ""
    await interaction.followup.send(f"Nouveau statut : {status.label()}{duration}.", ephemeral=True)


MEMORY_OFF = "La mémoire des membres est désactivée sur ce bot (MEMORY=off)."
MEMORY_BROKEN = ("La mémoire des membres est coupée pour l'instant, à cause d'une erreur du bot. "
                 "Préviens l'administrateur du bot.")


def memory_off(bot: ChatBot) -> str:
    """Pourquoi /memoire est indisponible : MEMORY=off, ou registre des membres illisible."""
    return MEMORY_BROKEN if bot.settings.memory else MEMORY_OFF


def bot_name(interaction: discord.Interaction) -> str:
    return interaction.guild.me.display_name if interaction.guild else interaction.client.user.name


@app_commands.command(name="memoire", description="Voir, modifier ou effacer ce que le bot a retenu sur toi")
@app_commands.describe(membre="Réservé aux administrateurs : la fiche d'un autre membre, ou son identifiant s'il est parti")
@app_commands.guild_only()
# discord.User plutôt que discord.Member : un membre parti du serveur reste accepté (Discord le résout par
# son identifiant), pour que sa fiche puisse encore être effacée.
async def memory_command(interaction: discord.Interaction, membre: discord.User | None = None) -> None:
    bot: ChatBot = interaction.client  # type: ignore[assignment]
    if bot.memory is None:
        await interaction.response.send_message(memory_off(bot), ephemeral=True)
        return
    target = membre or interaction.user
    if target.bot:
        await interaction.response.send_message("Les bots n'ont pas de fiche.", ephemeral=True)
        return
    if target.id != interaction.user.id and not interaction.permissions.administrator:
        logger.info("/memoire sur %s refusé pour %s", target, interaction.user)
        await interaction.response.send_message(
            "Seuls les administrateurs peuvent voir la fiche d'un autre membre.", ephemeral=True)
        return
    await see_user(bot.members, target)  # un membre qui n'a encore jamais écrit peut déjà activer la mémoire
    if target.id != interaction.user.id:
        # Lecture tracée, sans le contenu : un administrateur ne peut pas lire les fiches sans laisser de trace.
        logger.info("/memoire : fiche de %s ouverte par %s", target, interaction.user)
    card = MemoryCard(bot.memory, member_target(bot.memory, target, interaction.user.id),
                      interaction.user.id, bot_name(interaction))
    await card.send(interaction)


@app_commands.command(name="memoire-serveur", description="Voir ou modifier le fichier général du bot sur le serveur")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)  # comme /reset
async def server_memory_command(interaction: discord.Interaction) -> None:
    if not await allowed(interaction):
        return
    bot: ChatBot = interaction.client  # type: ignore[assignment]
    if bot.memory is None:
        await interaction.response.send_message(memory_off(bot), ephemeral=True)
        return
    await MemoryCard(bot.memory, server_target(), interaction.user.id, bot_name(interaction)).send(interaction)
