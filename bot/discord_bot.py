"""Le bot Discord : événements, réponses et réactions (commandes slash dans commands.py)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import discord
from discord import app_commands
from discord.ext import commands

from . import VERSION
from .attachments import (FILES_QUEUE_FULL, MAX_IMAGES, AttachmentReader, AttachmentRefused, HeldFiles,
                          attachments_marker, looks_like_image, looks_like_pdf, looks_like_text, readable_size,
                          quoted_text_prompt, text_budget, text_file_marker, text_file_prompt)
from .claude_client import ClaudeClient, ClaudeError
from .code_files import outgoing, prepare
from .commands import memory_command, reset_command, server_memory_command, skip_command
from .config import Settings, memory_warning
from .emojis import discord_emojis, emoji_names, short_emojis, usable_emojis
from .errors import DISCORD_FAILURES
from .files import write_atomic
from .gifs import EMBED_WAIT, gif_embed, looks_like_gif_link
from .history import ConversationHistory, render_line
from .images import ImageError
from .media import ImageData, fit_in_request
from .members import MemberRegistry, with_username
from .memory import DATE_FORMAT, MemoryStore
from .memory_views import reply_error, see_user
from .models import display_name, is_known, thinking_off
from .own_messages import OwnMessages
from .pdf_files import PdfError
from .presence import PresenceManager, Status
from .spontaneous import CHANCE, RECENT_MESSAGES, SpontaneousLog, finished_activity, is_silence, occasion_note
from .text_files import TEXT_FILE_MAX_BYTES, TextFileError

logger = logging.getLogger("chatbot.discord")

USER_MENTION = re.compile(r"<@!?(\d+)>")
# Couleur du message de démarrage : le vert du statut « en ligne » de Discord.
ONLINE_COLOR = discord.Color(0x23A55A)
# Longueur maximale du texte d'un message cité ou transféré transmis à Claude.
QUOTE_MAX_LENGTH = 500
# Messages écrits par quelqu'un ; les autres types sont des messages système
# (arrivée d'un membre, message épinglé, boost, fil créé…).
USER_MESSAGE_TYPES = (discord.MessageType.default, discord.MessageType.reply)
# En passant absent ou en « Ne pas déranger », le bot prévient son salon si on y a parlé depuis moins de :
DEPARTURE_WINDOW = timedelta(minutes=3)
# Messages de bots auxquels le bot répond d'affilée dans un salon, sans message d'un membre entre deux.
MAX_BOT_REPLIES = 5
# Passé ce délai sans message d'un bot dans le salon, le compte repart de zéro.
BOT_STREAK_RESET = timedelta(minutes=10)
UNEXPECTED_ERROR = "Désolé, une erreur inattendue est survenue."
# Messages supprimés retenus (voir forget_deleted) : bien plus que ceux qui peuvent attendre une réponse en même temps.
DELETED_KEPT = 1000


async def warn_authors(reply: Callable[[discord.Message, str], Awaitable[None]],
                       messages: list[discord.Message], text: str) -> None:
    """Un appel à Claude a échoué : ces messages n'auront pas de réponse. Chaque auteur est prévenu, une
    seule fois, sur son dernier message du lot : il sait qu'il doit le renvoyer."""
    for message in {m.author.id: m for m in messages}.values():
        await reply(message, text)


def author_name(user: discord.User | discord.Member) -> str:
    """Nom d'un auteur pour Claude, avec son nom d'utilisateur (voir with_username)."""
    return with_username(user.display_name, user.name)


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
    except DISCORD_FAILURES as exc:
        logger.warning("Indicateur « en train d'écrire » impossible dans #%s : %s", channel, exc)
        indicator = None
    try:
        yield
    finally:
        if indicator is not None:
            await indicator.__aexit__(None, None, None)


@dataclass
class PendingMessage:
    """Message du salon dédié, prêt à être envoyé à Claude."""
    message: discord.Message
    line: dict   # ligne conservée en mémoire (voir history.render_line)
    extra: str   # ajouté pour Claude seulement : explication du GIF, contenu des fichiers texte joints
    images: list[ImageData]  # images (ou la grille d'un GIF), ou un seul PDF

    @property
    def held_size(self) -> int:
        """Place que ses fichiers occupent en mémoire pendant l'attente (voir HeldFiles)."""
        return sum(len(image.data) for image in self.images) + len(self.extra)

    @property
    def user_text(self) -> str:
        """Texte envoyé à Claude : les images ou le fichier eux-mêmes remplacent leur repère."""
        line = {key: value for key, value in self.line.items() if key != "attachments"}
        return render_line(line) + self.extra


class BotStreaks:
    """Messages de bots reçus d'affilée, par salon. Deux bots qui se répondent tourneraient sans fin,
    avec un appel à Claude à chaque tour : au-delà de MAX_BOT_REPLIES, le bot ne leur répond plus."""

    def __init__(self) -> None:
        self._streaks: dict[int, tuple[int, datetime]] = {}  # salon : (messages, date du dernier)

    def human_spoke(self, channel_id: int) -> None:
        self._streaks.pop(channel_id, None)

    def count(self, channel_id: int, when: datetime) -> int:
        """Compte un message de bot adressé au bot et renvoie son rang dans la série (1 pour le premier).
        Un message ignoré compte aussi : un bot qui relance sans cesse reste ignoré."""
        count, last = self._streaks.get(channel_id, (0, when))
        if when - last > BOT_STREAK_RESET:
            count = 0
        self._streaks[channel_id] = (count + 1, when)
        return count + 1


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
        self.attachments = AttachmentReader(settings)
        self.history = ConversationHistory(settings.history_file, settings.max_history_length)
        self._pending: dict[int, list[PendingMessage]] = {}  # messages en attente de réponse, par salon
        self._preparing: dict[int, set[int]] = {}  # messages dont les pièces jointes se téléchargent, par salon
        self._deleted: set[int] = set()  # derniers messages supprimés, tous salons (voir forget_deleted)
        self._bot_streaks = BotStreaks()
        self._announced = False
        self.config_error: str | None = None  # salon dédié inutilisable : main.py sort en erreur
        self._memory_warning = memory_warning(settings) if self.memory else None
        self.presence = PresenceManager(self, self.claude, settings) if settings.presence else None
        # Messages spontanés (voir spontaneous.py) : ils naissent des statuts, rien sans eux.
        self.spontaneous = None
        if self.presence and settings.spontaneous_per_day:
            self.spontaneous = SpontaneousLog(settings.history_file.with_name("spontaneous_messages.json"),
                                              settings.spontaneous_per_day, settings.presence_speed)
        # Absent ou en « Ne pas déranger » : réponses différées, par salon.
        self._waiters: dict[int, asyncio.Task] = {}  # attente en cours
        self._mentions: dict[int, list[discord.Message]] = {}  # mentions en attente (hors salon dédié)
        self.held_files = HeldFiles()  # place occupée par les fichiers de tous les messages en attente
        # Messages en attente, gardés sur le disque pour y répondre même après un redémarrage.
        self._waiting_file = settings.history_file.with_name("waiting_messages.json")
        self._waiting: set[tuple[int, int]] = set()  # (salon, message)
        self._to_replay: set[tuple[int, int]] = set()  # messages à reprendre une fois connecté (voir load_waiting)
        self._replayed: set[tuple[int, int]] = set()  # messages repris au démarrage (voir remember_queued)
        self._stopping = False  # arrêt du bot en cours (voir forget_waiting)
        self._last_main_message: datetime | None = None  # date du dernier message reçu dans le salon dédié
        # Empreinte des commandes slash envoyées à Discord (voir sync_commands).
        self._commands_file = settings.history_file.with_name("slash_commands.sha256")

    # ------------------------------------------------------------ cycle de vie

    async def setup_hook(self) -> None:
        # Exécuté une seule fois au démarrage (contrairement à on_ready).
        self.load_waiting()
        if problem := await self.claude.check_access():
            # Sinon le bot s'annoncerait « En ligne » puis échouerait sur chaque message.
            logger.critical("Erreur de configuration : %s", problem)
            self.config_error = problem
            await self.close()  # connect() voit le bot fermé et rend la main à main.py
            return
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
        thinking = "on" if s.thinking else "off"
        if not s.thinking and thinking_off(s.model) is None:
            # Ce modèle réfléchit toujours : seul EFFORT en règle la quantité.
            thinking = "impossible à couper sur ce modèle (règle EFFORT)"
        logger.info("Modèle %s — réflexion : %s, effort : %s, recherche web : %s, mémoire des membres : %s",
                    s.model, thinking, s.effort or "par défaut",
                    "on" if s.web_search else "off", memory)
        logger.info("MAX_TOKENS : %s, API_TIMEOUT : %g s, journal : %s (niveau %s)", s.max_tokens, s.api_timeout,
                    s.log_file.resolve() if s.log_file else "console seulement", s.log_level)
        logger.info("Historique : %s, personnalité : %s", s.history_file.resolve(), s.personality_file.resolve())
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
        if s.spontaneous_per_day and not self.presence:
            logger.warning("SPONTANEOUS_PER_DAY est ignoré : les messages spontanés nécessitent PRESENCE=on.")
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
        except DISCORD_FAILURES as exc:  # le bot fonctionne quand même ; nouvel essai au prochain démarrage
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
        except DISCORD_FAILURES as exc:
            logger.error("Impossible d'envoyer le message de démarrage : %s", exc)

    async def close(self) -> None:
        self._stopping = True  # voir forget_waiting
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
        # ou réponse à l'un de ses messages. Deux bots peuvent ainsi discuter ensemble, mais pas sans fin.
        if not message.author.bot:
            self._bot_streaks.human_spoke(message.channel.id)
        else:
            listed = self.user in message.mentions
            addressed = listed or await self.replies_to_me(message)
            # Une mention dans le texte (<@id>) devrait figurer dans la liste des mentions fournie par Discord :
            # le journal dit si c'est le cas, pour comprendre pourquoi un bot qui mentionne le bot est ignoré.
            in_text = self.user is not None and self.user.id in message.raw_mentions
            if in_text:
                logger.info("%s mentionne le bot dans #%s (message %s) : %s.", message.author, message.channel,
                            message.id, "mention transmise par Discord" if listed else
                            "mention absente de la liste de Discord" + ("" if addressed else ", message ignoré"))
            if not addressed:
                return
            if (rank := self._bot_streaks.count(message.channel.id, message.created_at)) > MAX_BOT_REPLIES:
                if rank == MAX_BOT_REPLIES + 1:
                    logger.warning("#%s : %s réponses à des bots d'affilée (dernier : %s). Le bot ne leur répond "
                                   "plus jusqu'au prochain message d'un membre, ou après %s minutes sans message de "
                                   "bot.", message.channel, MAX_BOT_REPLIES, message.author,
                                   BOT_STREAK_RESET.seconds // 60)
                elif in_text:
                    logger.info("Message %s ignoré : limite de réponses à des bots d'affilée (rang %s).",
                                message.id, rank)
                return

        in_main_channel = message.channel.id == self.settings.channel_id
        # Noté dès l'arrivée, avant l'enregistrement des membres (qui peut lire le message cité) : un message spontané
        # préparé pendant ce temps voit déjà que quelqu'un a écrit (voir speak_up).
        if in_main_channel and (self._last_main_message is None or message.created_at > self._last_main_message):
            self._last_main_message = message.created_at
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
        # Ailleurs dans leur traitement (registre des membres, préparation, appel à Claude), le bot les retrouve ici
        # et ne leur répond pas. Les identifiants de Discord croissent avec le temps : les plus petits sont oubliés.
        self._deleted |= message_ids
        if len(self._deleted) > DELETED_KEPT:
            self._deleted = set(sorted(self._deleted)[-DELETED_KEPT:])
        self.held_files.release(*message_ids)
        keys = {(channel_id, message_id) for message_id in message_ids}
        if keys & self._waiting:
            self._waiting -= keys
            self._save_waiting()
        if await self.history.delete(channel_id, message_ids):
            logger.info("%s message(s) supprimé(s) dans le salon %s : mémoire mise à jour.", len(message_ids), channel_id)

    async def handle_main_channel(self, message: discord.Message) -> None:
        content = message.content.strip()
        # Place réservée avant tout téléchargement, et jusqu'à la réponse (voir answer_main_channel).
        if not self.held_files.reserve(message.id, readable_size(message.attachments)):
            logger.warning("Trop de fichiers en attente : message %s de %s refusé.", message.id, message.author)
            await self.safe_reply(message, FILES_QUEUE_FULL)
            return
        # Tant qu'il se prépare, les messages suivants l'attendent (voir answer_main_channel) : une fois prêt,
        # ou refusé, c'est lui qui lance leur réponse, quoi qu'il arrive.
        channel_id = message.channel.id
        arrival = asyncio.get_running_loop().time()
        self.remember_queued(message)
        self._preparing.setdefault(channel_id, set()).add(message.id)
        pending = None
        try:
            pending = await self.read_main_message(message, content)
        except Exception:
            logger.exception("Erreur inattendue pendant la lecture d'un message")
            await self.safe_reply(message, UNEXPECTED_ERROR)
        finally:
            self._preparing[channel_id].discard(message.id)
            if message.id in self._deleted:  # supprimé depuis son arrivée (voir forget_deleted)
                logger.info("Message %s supprimé avant sa réponse : pas de réponse.", message.id)
                pending = None
        if pending is None:
            self.held_files.release(message.id)
            self.forget_waiting(message)
            if not self._pending.get(channel_id):
                return
        else:
            self.held_files.settle(message.id, pending.held_size)
            # Un seul appel à la fois par salon : l'historique reste cohérent. Les messages arrivés
            # pendant une réponse attendent dans la file, puis partent ensemble dans un seul appel
            # (ou plusieurs, s'ils sont trop lourds ensemble : voir fit_in_request).
            self._pending.setdefault(channel_id, []).append(pending)
        # Courte pause, comptée depuis l'arrivée du message : tout ce qui arrive entre-temps part dans la même
        # réponse. Une image, plus longue à envoyer, arrive souvent après le texte qui l'accompagne.
        if self.available() and channel_id not in self._waiters:
            await asyncio.sleep(max(0, self.settings.reply_pause - (asyncio.get_running_loop().time() - arrival)))
        if not self.available() or channel_id in self._waiters:
            # Tous ceux de la file, pas seulement ce message : refusé, il a peut-être été attendu par d'autres ;
            # parti entre-temps avec une autre réponse, il n'a plus à attendre.
            for queued in self._pending.get(channel_id, []):
                self.answer_later(queued.message, self.answer_main_channel)
            return
        await self.answer_main_channel(message.channel)

    async def read_main_message(self, message: discord.Message, content: str) -> PendingMessage | None:
        """Télécharge ce qui accompagne un message du salon dédié et le prépare pour Claude, ou renvoie None
        s'il n'y a rien à lire ou si une pièce jointe est refusée (l'auteur a alors été prévenu)."""
        forward, forward_images = await self.read_forward(message)
        if not (content or message.attachments or message.stickers or forward):
            return None  # sondage, transfert d'un message vide… : rien à lire

        attachments = await self.read_attachments(message)
        if attachments is False:
            return None  # pièce jointe refusée, l'utilisateur a été prévenu
        images, text_files = attachments
        gif = None if images else await self.read_gif(message)
        if gif:
            images = [gif[0]]

        # Les images du message cité ne sont utilisées que si le message n'en contient pas lui-même. Ses fichiers
        # texte se partagent la limite de taille avec ceux du message.
        reply, reply_images, reply_texts = await self.read_reply(
            message, want_images=not (images or forward_images), text_budget=text_budget(message.attachments))
        images = images or forward_images or reply_images

        line: dict = {"id": message.id, "date": self.timestamp(message), "author": author_name(message.author)}
        if forward:
            line["forward"] = forward
        else:
            if reply:
                line["reply_to"] = message.reference.message_id  # citation retirée si ce message est supprimé
                line["reply"] = reply
            line["content"] = self.message_body(message)
        markers = [gif[2] if gif else attachments_marker(images)] if images else []
        extra = gif[1] if gif else ""
        for name, text in text_files:
            extra += text_file_prompt(name, text)
            markers.append(text_file_marker(name))
        for name, text in reply_texts:
            extra += quoted_text_prompt(name, text)
        if markers:
            line["attachments"] = markers
        return PendingMessage(message, line, extra, images)

    async def answer_main_channel(self, channel: discord.abc.Messageable, late: Status | None = None) -> None:
        """Répond aux messages en attente du salon dédié : en une fois, ou en plusieurs réponses à la suite
        s'ils ne tiennent pas dans une seule requête."""
        pending = self._pending.setdefault(channel.id, [])
        split = False  # réponse découpée : chaque morceau cite son dernier message
        # Jusqu'au dernier message : une réponse différée est la seule tâche pour tous ceux de son attente.
        while True:
            batch: list[PendingMessage] = []
            try:
                async with self.history.lock(channel.id):
                    # Tous les messages attendent ceux encore en préparation : sinon le bot répondrait à « tu en
                    # penses quoi ? » sans avoir vu l'image envoyée avec. Le message en préparation relancera la
                    # réponse.
                    if not pending or self._preparing.get(channel.id):
                        return
                    pending.sort(key=lambda p: p.message.id)
                    count = fit_in_request([p.images for p in pending], [p.user_text for p in pending])
                    if count < len(pending):
                        split = True
                        logger.info("Messages trop lourds pour une seule réponse dans #%s : %s message(s) sur %s "
                                    "d'abord, les autres ensuite.", channel, count, len(pending))
                    batch = pending[:count]
                    del pending[:count]
                    try:  # ces messages ont quitté la file : le finally libère leur place, quoi qu'il arrive
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
                        async with typing(channel):
                            try:
                                member_notes = await self.member_notes([p.message for p in batch])
                                answer = await self.claude.ask(self.history.get(channel.id), parts,
                                                               emojis=emoji_names(last.guild),
                                                               member_notes=member_notes,
                                                               own_messages=self.own_messages(channel))
                            except ClaudeError as exc:
                                await warn_authors(self.safe_reply, [p.message for p in batch], exc.user_message)
                                continue
                            except Exception:
                                logger.exception("Erreur inattendue pendant le traitement d'un message")
                                await warn_authors(self.safe_reply, [p.message for p in batch], UNEXPECTED_ERROR)
                                continue
                    finally:
                        self.held_files.release(*(p.message.id for p in batch))
                    # Supprimés pendant que Claude écrivait : sans eux, la réponse n'a plus lieu d'être. Si d'autres
                    # restent, elle part ; la suppression attend ce verrou, puis efface ces messages de la mémoire.
                    kept = [p.message for p in batch if p.message.id not in self._deleted]
                    if not kept:
                        logger.info("Message(s) supprimé(s) dans #%s pendant la réponse : elle n'est pas envoyée.",
                                    channel)
                        continue
                    # Les longs blocs de code, envoyés en fichier joint, n'y sont que résumés (voir code_files.py).
                    memory = short_emojis(prepare(answer.text).memory)
                    await self.history.add_exchange(channel.id, lines, memory, answer.web_used)
                await self.send_answer(kept[-1], answer.text, reply=split)
            finally:
                if batch:
                    self.forget_waiting(*(p.message for p in batch))

    async def handle_mention(self, message: discord.Message) -> None:
        # Comme dans le salon dédié : les mentions arrivées pendant une réponse attendent dans la file,
        # puis partent ensemble dans un seul appel (ou plusieurs, si elles sont trop lourdes ensemble).
        # Leurs fichiers sont téléchargés au moment de répondre : leur place est réservée dès maintenant.
        if not self.held_files.reserve(message.id, readable_size(message.attachments)):
            logger.warning("Trop de fichiers en attente : mention %s de %s refusée.", message.id, message.author)
            await self.safe_reply(message, FILES_QUEUE_FULL)
            return
        self.remember_queued(message)
        self._mentions.setdefault(message.channel.id, []).append(message)
        if not self.available() or message.channel.id in self._waiters:
            self.answer_later(message, self.answer_mentions)
            return
        await self.answer_mentions(message.channel)

    async def answer_mentions(self, channel: discord.abc.Messageable, late: Status | None = None) -> None:
        """Répond en une fois aux mentions en attente d'un salon (hors salon dédié)."""
        # Un seul appel à la fois par salon, envoi compris : la réponse suivante voit la précédente dans le contexte.
        async with self.history.lock(channel.id):
            batch = sorted(self._mentions.pop(channel.id, []), key=lambda m: m.id)
            if not batch:
                return  # déjà parties avec une mention précédente
            if len(batch) > 1:
                logger.info("%s mentions regroupées en une seule réponse dans #%s", len(batch), channel)
            try:
                await self.answer_mention(batch, late)
            finally:
                self.forget_waiting(*batch)
                self.held_files.release(*(m.id for m in batch))

    async def answer_mention(self, mentions: list[discord.Message], late: Status | None = None) -> None:
        """Hors du salon dédié : répond aux mentions d'un même salon en lisant les messages récents, en une
        fois, ou en plusieurs réponses à la suite si elles ne tiennent pas dans une seule requête."""
        channel = mentions[-1].channel
        answered, parts = [], []
        for message in mentions:
            try:
                part = await self.mention_part(message)
            except Exception:
                logger.exception("Erreur inattendue pendant la lecture d'une mention")
                await self.safe_reply(message, UNEXPECTED_ERROR)
                continue
            if part and message.id not in self._deleted:  # pas supprimée depuis son arrivée
                answered.append(message)
                parts.append(part)
        # Mentions à pièce jointe refusée (leurs auteurs ont été prévenus) ou supprimées : hors du contexte.
        refused = {m.id for m in mentions} - {m.id for m in answered}
        sent: list[str] = []  # en retard, réponses déjà envoyées (postées après la fin du contexte)
        split = False  # réponse découpée : chaque morceau cite sa dernière mention
        # Liste vide : pièces jointes refusées, les auteurs ont été prévenus.
        while parts:
            count = fit_in_request([images for _, images in parts], [text for text, _ in parts])
            if count < len(parts):
                split = True
                logger.info("Mentions trop lourdes pour une seule réponse dans #%s : %s mention(s) sur %s "
                            "d'abord, les autres ensuite.", channel, count, len(parts))
            group, group_parts = answered[:count], parts[:count]
            del answered[:count], parts[:count]
            last = group[-1]

            async with typing(channel):
                try:
                    # Contexte jusqu'à maintenant, relu pour chaque réponse : elle voit celle que le bot vient
                    # d'envoyer (postée après ces mentions). En retard, jusqu'à la dernière mention : la discussion
                    # qui a suivi recouvrirait celle-ci ; les réponses déjà envoyées sont alors ajoutées à la fin.
                    # Seules les mentions pas encore traitées sont décrites à part : les autres sont des messages
                    # normaux.
                    pending = refused | {m.id for m in group} | {m.id for m in answered}
                    lines = await self.mention_context(channel, pending, before=mentions[-1] if late else None) + sent
                    channel_name = f"#{channel.name}" if hasattr(channel, "name") else "un message privé"
                    header = []
                    if late:
                        header.append(self.presence.late_note(late, several=len(group_parts) > 1))
                    if note := self.presence_note():
                        header.append(note)
                    if lines:
                        header.append(f"Contexte : derniers messages dans {channel_name}\n" + "\n".join(lines))
                    text, images = group_parts[0]
                    group_parts[0] = ("\n\n".join([*header, text]), images)
                    answer = await self.claude.ask([], group_parts, emojis=emoji_names(last.guild), cache_history=False,
                                                   member_notes=await self.member_notes(group),
                                                   own_messages=self.own_messages(channel, remembered=False))
                except ClaudeError as exc:
                    await warn_authors(self.safe_reply, group, exc.user_message)
                    continue
                except Exception:
                    logger.exception("Erreur inattendue pendant le traitement d'une mention")
                    await warn_authors(self.safe_reply, group, UNEXPECTED_ERROR)
                    continue
            # Toutes supprimées pendant que Claude écrivait : la réponse n'a plus lieu d'être.
            if not (kept := [m for m in group if m.id not in self._deleted]):
                logger.info("Mention(s) supprimée(s) dans #%s pendant la réponse : elle n'est pas envoyée.", channel)
                continue
            last = kept[-1]
            # En retard ou découpée, la réponse cite la dernière mention : on sait à quoi elle répond.
            await self.send_answer(last, answer.text, reply=late is not None or split)
            if late:
                me = last.guild.me if last.guild else self.user
                sent.append(f"{author_name(me)} : {short_emojis(prepare(answer.text).memory)}")

    async def mention_context(self, channel: discord.abc.Messageable, mention_ids: set[int],
                              before: discord.Message | None) -> list[str]:
        """Derniers messages du salon avant before (ou jusqu'à maintenant), du plus ancien au plus récent,
        sans les mentions auxquelles le bot répond : elles sont décrites à part, avec leurs pièces jointes."""
        lines: list[str] = []
        if self.settings.context_message_limit > 0:
            try:
                async for msg in channel.history(limit=self.settings.context_message_limit, before=before):
                    if msg.id in mention_ids:
                        continue
                    content = short_emojis(msg.clean_content.strip())
                    if msg.attachments:
                        content = (content + " [pièce jointe]").strip()
                    if msg.stickers:
                        content = (content + " [sticker]").strip()
                    if content:
                        lines.append(f"{author_name(msg.author)} : {content}")
            except discord.Forbidden:
                logger.warning("Pas la permission de lire l'historique de #%s.", channel)
            except DISCORD_FAILURES as exc:
                # Le contexte n'est qu'un complément : la panne ne prive pas la personne de sa réponse.
                logger.warning("Derniers messages de #%s illisibles, réponse sans ce contexte : %s", channel, exc)
        lines.reverse()
        return lines

    async def mention_part(self, message: discord.Message) -> tuple[str, list[ImageData]] | None:
        """Une mention telle qu'elle est transmise à Claude, ou None si une pièce jointe est refusée
        (l'auteur a été prévenu)."""
        # Seuls les MAX_IMAGES premières images (à défaut, le premier PDF) et les fichiers texte de la
        # mention sont lus (les autres pièces jointes sont ignorées). Les images du message cité
        # ne servent que s'il n'y a ni image ni PDF.
        image_files = [a for a in message.attachments if looks_like_image(a)][:MAX_IMAGES]
        pdfs = [a for a in message.attachments if looks_like_pdf(a)] if self.settings.pdf_max_pages else []
        try:
            if image_files:
                images = await self.attachments.download_images(image_files)
            else:
                images = [await self.attachments.download_pdf(pdfs[0])] if pdfs else []
        except (ImageError, PdfError) as exc:
            await self.safe_reply(message, exc.user_message)
            return None
        gif = None if images else await self.read_gif(message)
        if gif:
            images = [gif[0]]
        reply, reply_images, reply_texts = await self.read_reply(message, want_images=not images,
                                                                 text_budget=text_budget(message.attachments))
        images = images or reply_images

        bot_name = f"@{message.guild.me.display_name}" if message.guild else f"@{self.user.name}"
        text = short_emojis(message.clean_content.replace(bot_name, "").strip()) or "(simple mention, sans texte)"
        reply_part = f" ({reply})" if reply else ""
        prompt = f"{self.timestamp(message)}, {author_name(message.author)} te mentionne{reply_part} et écrit : {text}"
        if gif:
            prompt += gif[1]
        for name, content in reply_texts:
            prompt += quoted_text_prompt(name, content)

        text_files = [a for a in message.attachments if looks_like_text(a)]
        if text_files:
            try:
                for name, content in await self.attachments.download_text_files(text_files):
                    prompt += text_file_prompt(name, content)
            except TextFileError as exc:
                await self.safe_reply(message, exc.user_message)
                return None
        return prompt, images

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

    async def on_status_change(self, before: Status, after: Status, skipped: bool) -> None:
        """skipped : statut abandonné avant sa fin avec /skip (voir PresenceManager.skip)."""
        if after.available:
            if not skipped:  # anime ou partie de jeu interrompu : il n'en a pas vu la fin, il n'en parle pas
                await self.speak_up(before, after)
        else:
            await self.announce_departure(before, after)

    async def announce_departure(self, before: Status, after: Status) -> None:
        """Passe absent ou en « Ne pas déranger » : prévient le salon dédié si une conversation y est en cours."""
        if not before.available or self._last_main_message is None:
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
                                                   emojis=emoji_names(channel.guild),
                                                   own_messages=self.own_messages(channel))
            except ClaudeError as exc:
                logger.warning("Message d'absence non envoyé : %s", exc.user_message)
                return
            lines = [{"content": text} for text, _ in parts]
            await self.history.add_exchange(channel.id, lines, short_emojis(prepare(answer.text).memory),
                                            answer.web_used)
        logger.info("Conversation en cours dans #%s : le salon est prévenu de l'absence.", channel)
        prepared = prepare(answer.text)
        for chunk, extra in outgoing(discord_emojis(short_emojis(prepared.text), channel.guild), prepared.files):
            try:
                await channel.send(chunk, **extra)
            except DISCORD_FAILURES as exc:
                logger.error("Envoi du message d'absence impossible : %s", exc)
                return

    async def speak_up(self, before: Status, after: Status) -> None:
        """Vient de finir son anime ou sa partie de jeu : en parle peut-être de lui-même dans le salon dédié, s'il
        est calme (voir spontaneous.py)."""
        if not self.spontaneous or not finished_activity(before, after) or random.random() >= CHANCE:
            return
        channel = self.get_channel(self.settings.channel_id)
        if channel is None:
            return

        def member_waiting() -> bool:
            """Un message du salon attend sa réponse (ou se prépare) : c'est à lui que le bot répond."""
            return bool(self._pending.get(channel.id) or self._preparing.get(channel.id) or channel.id in self._waiters
                        or self._last_main_message != seen)

        seen = self._last_main_message  # un message arrivé depuis, même pas encore en attente, l'emporte
        async with self.history.lock(channel.id):
            # Vérifié sous le verrou : une réponse vient peut-être de partir, ou un message attend la sienne.
            if self.presence.status is not after or member_waiting():
                return
            # Seuls les messages écrits par des membres comptent : ni le message « En ligne » du démarrage, ni les
            # autres bots, ni les annonces de Discord (message épinglé, arrivée d'un membre…) ne répondent au bot ou ne
            # troublent le calme du salon.
            try:
                # Du plus récent au plus ancien.
                members = [m async for m in channel.history(limit=RECENT_MESSAGES)
                           if not m.author.bot and m.type in USER_MESSAGE_TYPES]
            except DISCORD_FAILURES as exc:
                logger.warning("Derniers messages de #%s illisibles, pas de message spontané : %s", channel, exc)
                return
            if not self.spontaneous.allowed(discord.utils.utcnow(), members[0].id if members else None):
                return
            note = occasion_note(before, datetime.now(self.settings.timezone))
            parts = [(text, []) for text in (self.presence_note(), note) if text]
            # Fiches des derniers membres à avoir écrit : il peut s'adresser à l'un d'eux que le sujet intéresse.
            notes = self.memory.notes_for(list(dict.fromkeys(m.author.id for m in members))) if self.memory else ""
            try:
                answer = await self.claude.ask(self.history.get(channel.id), parts, emojis=emoji_names(channel.guild),
                                               member_notes=notes, own_messages=self.own_messages(channel))
            except ClaudeError as exc:
                logger.warning("Message spontané non envoyé : %s", exc.user_message)
                return
            if is_silence(answer.text):
                logger.info("Fin de « %s » : rien à dire dans #%s, pas de message spontané.", before.title, channel)
                return
            if member_waiting():  # un membre a écrit pendant que Claude écrivait : le salon n'est plus calme
                logger.info("Fin de « %s » : un membre vient d'écrire dans #%s, message spontané abandonné.",
                            before.title, channel)
                return
            sent = None
            prepared = prepare(answer.text)
            for chunk, extra in outgoing(discord_emojis(short_emojis(prepared.text), channel.guild), prepared.files):
                try:
                    sent = await channel.send(chunk, **extra)
                except DISCORD_FAILURES as exc:
                    logger.error("Envoi du message spontané impossible : %s", exc)
                    break
            if sent is None:
                return
            self.spontaneous.record(sent.id, discord.utils.utcnow())
            lines = [{"content": text} for text, _ in parts]
            await self.history.add_exchange(channel.id, lines, short_emojis(prepared.memory), answer.web_used)
        logger.info("Fin de « %s » : message spontané dans #%s.", before.title, channel)

    def remember_waiting(self, message: discord.Message) -> None:
        key = (message.channel.id, message.id)
        if key not in self._waiting:  # déjà dans le fichier : inutile de le réécrire (voir handle_main_channel)
            self._waiting.add(key)
            self._save_waiting()

    def remember_queued(self, message: discord.Message) -> None:
        """Message en cours de traitement : gardé sur le disque jusqu'à sa réponse, pour un redémarrage. Sauf s'il
        est repris au démarrage : s'il faisait tomber le bot (image énorme…), il recommencerait à chaque reprise."""
        if (message.channel.id, message.id) not in self._replayed:
            self.remember_waiting(message)

    def forget_waiting(self, *messages: discord.Message) -> None:
        if self._stopping:
            return  # arrêt du bot : sa réponse en cours est coupée, le message sera repris au redémarrage
        keys = {(m.channel.id, m.id) for m in messages}
        if keys & self._waiting:
            self._waiting -= keys
            self._save_waiting()

    def _save_waiting(self) -> None:
        try:
            # Écrit à chaque message, dans la boucle du bot : sans fsync, qui le figerait sur un volume lent.
            write_atomic(self._waiting_file, json.dumps(sorted(self._waiting)), flush_to_disk=False)
        except OSError as exc:
            logger.warning("Impossible d'écrire %s : %s", self._waiting_file, exc)

    def load_waiting(self) -> None:
        """Avant la connexion à Discord : lit les messages restés sans réponse avant le redémarrage. Gardés en
        mémoire jusqu'à leur reprise : un message reçu entre-temps réécrit le fichier, qui doit encore les contenir."""
        try:
            saved = json.loads(self._waiting_file.read_text(encoding="utf-8"))
            keys = {(channel_id, message_id) for channel_id, message_id in saved}
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("%s illisible, ignoré : %s", self._waiting_file, exc)
            return
        self._to_replay = keys
        self._waiting |= keys

    async def replay_waiting(self) -> None:
        """Une fois connecté : reprend les messages lus par load_waiting. Ceux reçus depuis le démarrage ne sont
        pas concernés : ils sont déjà en cours de traitement."""
        keys, self._to_replay = self._to_replay, set()
        if not keys:
            return
        messages = []
        unread = set()  # Discord en panne, réseau coupé : pas une raison de les perdre
        for channel_id, message_id in sorted(keys):
            try:
                channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
                messages.append(await channel.fetch_message(message_id))
            except (discord.NotFound, discord.Forbidden) as exc:  # message supprimé, salon inaccessible
                logger.info("Message %s du salon %s non repris : %s", message_id, channel_id, exc)
            except DISCORD_FAILURES as exc:
                logger.warning("Message %s du salon %s pas relu, gardé pour le prochain démarrage : %s",
                               message_id, channel_id, exc)
                unread.add((channel_id, message_id))
        keys -= unread
        self._waiting -= keys  # pas les messages arrivés pendant les téléchargements, toujours en attente
        self._replayed |= keys  # un seul essai pour ceux-là (voir remember_queued)
        self._save_waiting()
        if messages:
            logger.info("%d message(s) resté(s) sans réponse avant le redémarrage : reprise.", len(messages))
            # En parallèle, comme à leur arrivée : ceux du salon dédié sont regroupés.
            ordered = sorted(messages, key=lambda m: m.id)
            results = await asyncio.gather(*(self.on_message(m) for m in ordered), return_exceptions=True)
            for message, result in zip(ordered, results):
                if isinstance(result, Exception):
                    logger.error("Reprise du message %d impossible", message.id, exc_info=result)

    # ------------------------------------------------------- pièces jointes

    async def read_attachments(
        self, message: discord.Message
    ) -> tuple[list[ImageData], list[tuple[str, str]]] | bool:
        """Pièces jointes du message (voir AttachmentReader.read), ou False si l'envoi est refusé :
        l'utilisateur est alors prévenu."""
        try:
            return await self.attachments.read(message.attachments)
        except (AttachmentRefused, ImageError, PdfError, TextFileError) as exc:
            await self.safe_reply(message, exc.user_message)
            return False

    async def read_gif(self, message: discord.Message) -> tuple[ImageData, str, str] | None:
        """GIF choisi dans le sélecteur de Discord (voir AttachmentReader.read_gif), ou None."""
        # L'aperçu d'un lien peut arriver juste après le message : Discord l'ajoute par une modification,
        # que discord.py reporte sur ce même message. Attendu un court moment, pour un lien de GIF seulement.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + EMBED_WAIT
        while not gif_embed(message.embeds) and looks_like_gif_link(message.content) and loop.time() < deadline:
            await asyncio.sleep(0.25)
        return await self.attachments.read_gif(message.embeds)

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
            except DISCORD_FAILURES as exc:
                logger.info("Message cité inaccessible : %s", exc)
                return None
            ref.resolved = cited  # évite de le télécharger une seconde fois
        return cited if isinstance(cited, discord.Message) else None  # None : message cité supprimé

    async def replies_to_me(self, message: discord.Message) -> bool:
        cited = await self.fetch_cited(message)
        return cited is not None and cited.author == self.user

    async def read_reply(self, message: discord.Message, want_images: bool, text_budget: int = TEXT_FILE_MAX_BYTES
                         ) -> tuple[str | None, list[ImageData], list[tuple[str, str]]]:
        """Lit le message auquel l'utilisateur répond.

        Renvoie sa description pour Claude, ses images si want_images est vrai, et (nom, contenu) de ses fichiers
        texte qui tiennent dans text_budget octets ; ou (None, [], []) si le message ne répond à rien ou si le
        message cité est inaccessible. Citer un fichier de code envoyé par le bot suffit ainsi à le lui faire
        relire, sans le renvoyer.
        """
        cited = await self.fetch_cited(message)
        if cited is None:
            return None, [], []

        images = await self.attachments.quoted_images(cited.attachments, "cité") if want_images else []
        texts, too_long = await self.attachments.quoted_text_files(cited.attachments, text_budget)
        description = self.describe_quoted(cited.clean_content, images, cited.attachments, cited.stickers,
                                           [name for name, _ in texts], too_long)

        who = "à ton message" if cited.author == self.user else f"au message de {author_name(cited.author)}"
        return f"en réponse {who} du {self.timestamp(cited)} : {description or '(message vide)'}", images, texts

    async def read_forward(self, message: discord.Message) -> tuple[str | None, list[ImageData]]:
        """Lit le message transféré (bouton « Transférer » de Discord).

        Renvoie sa description pour Claude et ses images, ou (None, []) si le message
        n'est pas un transfert ou si le message transféré ne contient rien de lisible.
        """
        if not message.message_snapshots:
            return None, []
        snapshot = message.message_snapshots[0]

        images = await self.attachments.quoted_images(snapshot.attachments, "transféré")
        # Contrairement à un message normal, le contenu transféré garde ses mentions brutes (<@123>).
        content = self.clean_mentions(snapshot.content)
        description = self.describe_quoted(content, images, snapshot.attachments, snapshot.stickers)
        if not description:
            return None, []
        return f"du {self.timestamp(snapshot)} : {description}", images

    @staticmethod
    def describe_quoted(content: str, images: list[ImageData], attachments: list, stickers: list,
                        texts: list[str] | None = None, too_long: list[str] | None = None) -> str:
        """Description d'un message cité ou transféré (vide s'il n'y a rien de lisible). texts : noms de ses
        fichiers texte lus ; too_long : ceux qui ne l'ont pas été, faute de place."""
        texts, too_long = texts or [], too_long or []
        content = short_emojis(content.strip())
        if len(content) > QUOTE_MAX_LENGTH:
            content = content[:QUOTE_MAX_LENGTH].rstrip() + "…"
        parts = [f"« {content} »"] if content else []
        if len(images) > 1:
            parts.append(f"[ses {len(images)} images sont jointes à ce message]")
        elif images:
            parts.append("[son image est jointe à ce message]")
        for name in texts:
            # Gardé dans la mémoire : vrai aussi quand elle relit ce message plus tard, sans le fichier.
            parts.append(f"[fichier texte « {name} » : tu le lis avec ce message seulement, son contenu n'est pas "
                         "conservé en mémoire]")
        for name in too_long:
            parts.append(f"[fichier texte « {name} » : trop long pour être relu avec ce message]")
        # Les autres pièces jointes (images non transmises, vidéos…) sont signalées, même à côté d'un fichier lu.
        described = set(texts) | set(too_long)
        if not images and any(getattr(a, "filename", None) not in described for a in attachments):
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

    def own_messages(self, channel: discord.abc.Messageable, remembered: bool = True) -> OwnMessages:
        """Outil « mes_messages » de Claude dans ce salon. remembered : salon dont le bot garde la mémoire
        (le salon dédié), qui prend alors le nouveau texte d'un message modifié. Toujours fourni à Claude :
        un outil présent à un appel et absent au suivant ferait réécrire tout le cache de prompt."""
        on_edit = None
        if remembered:
            async def on_edit(old: str, new: str) -> None:
                await self.history.revise_reply(channel.id, old, new)
        return OwnMessages(channel, self.user, on_edit)

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
        # Les longs blocs de code partent en fichier joint (voir code_files.py).
        prepared = prepare(answer)
        text = discord_emojis(short_emojis(prepared.text), message.guild)
        for i, (chunk, extra) in enumerate(outgoing(text, prepared.files)):
            # fail_if_not_exists=False : si le message a été supprimé entre-temps, envoi normal.
            reference = message.to_reference(fail_if_not_exists=False) if reply and i == 0 else None
            try:
                await message.channel.send(chunk, reference=reference, mention_author=False, **extra)
            except DISCORD_FAILURES as exc:
                logger.error("Envoi de la réponse impossible : %s", exc)
                return

    async def safe_reply(self, message: discord.Message, text: str) -> None:
        """Prévient l'auteur d'un message en y répondant ; si Discord refuse, à la suite dans le salon. Sauf si le
        message a été supprimé : seul dans le salon, l'avertissement ne se rapporterait à rien."""
        try:
            await message.reply(text, mention_author=False)
            return
        except DISCORD_FAILURES:
            pass
        try:
            await message.channel.fetch_message(message.id)
        except discord.NotFound:
            logger.info("Message %s supprimé : avertissement non envoyé.", message.id)
            return
        except DISCORD_FAILURES:
            pass  # existence inconnue (Discord injoignable…) : l'avertissement part quand même
        try:
            await message.channel.send(text)
        except DISCORD_FAILURES as exc:
            logger.error("Impossible d'envoyer un message : %s", exc)
