"""Le bot Discord : événements, commandes et réactions."""

from __future__ import annotations

import asyncio
import logging
import random
import re

import discord
from discord import app_commands
from discord.ext import commands

from . import VERSION
from .claude_client import ClaudeClient, ClaudeError, ImageData
from .config import Settings
from .history import IMAGE_MARKER, ConversationHistory
from .images import ImageError, prepare_image
from .text_utils import split_message

logger = logging.getLogger("chatbot.discord")

CUSTOM_EMOJI_ONLY = re.compile(r"^<a?:\w+:\d+>$")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff")
# Longueur maximale du texte d'un message cité transmis à Claude.
QUOTE_MAX_LENGTH = 500


def looks_like_image(attachment: discord.Attachment) -> bool:
    return (attachment.content_type or "").startswith("image/") or attachment.filename.lower().endswith(IMAGE_EXTENSIONS)


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
        self._announced = False

    # ------------------------------------------------------------ cycle de vie

    async def setup_hook(self) -> None:
        # Exécuté une seule fois au démarrage (contrairement à on_ready).
        self.tree.add_command(reset_command)
        synced = await self.tree.sync()
        logger.info("%d commande(s) slash synchronisée(s).", len(synced))

    async def on_ready(self) -> None:
        logger.info("Connecté en tant que %s (version %s)", self.user, VERSION)
        if self._announced:
            return  # simple reconnexion : on ne reposte pas le message de démarrage
        self._announced = True

        channel = self.get_channel(self.settings.channel_id)
        if channel is None:
            logger.error("Salon %s introuvable : vérifie CHANNEL_ID et les permissions du bot.",
                         self.settings.channel_id)
            return
        name = channel.guild.me.display_name if getattr(channel, "guild", None) else self.user.name
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
        await self.claude.close()
        await super().close()

    # ------------------------------------------------------------- messages

    async def on_message(self, message: discord.Message) -> None:
        # Ignore les bots (dont lui-même) et les webhooks : évite les boucles infinies.
        if message.author.bot or message.webhook_id is not None:
            return

        in_main_channel = message.channel.id == self.settings.channel_id
        # Mention directe uniquement (@everyone et @here ne comptent pas).
        mentioned = self.user is not None and self.user in message.mentions

        if in_main_channel:
            await self.handle_main_channel(message)
        elif mentioned:
            await self.handle_mention(message)

    async def handle_main_channel(self, message: discord.Message) -> None:
        if await self.reply_to_sticker(message):
            return
        if await self.reply_to_emoji(message):
            return

        image = await self.read_image(message)
        if image is False:
            return  # pièce jointe refusée, l'utilisateur a été prévenu

        # L'image du message cité n'est utilisée que si le message n'en contient pas lui-même.
        reply, reply_image = await self.read_reply(message, want_image=image is None)
        image = image or reply_image

        text = message.clean_content.strip() or "(message sans texte)"
        reply_part = f" ({reply})" if reply else ""
        user_text = f"{self.timestamp(message)}, {message.author.display_name} a écrit{reply_part} : {text}"
        stored_text = user_text + (f" {IMAGE_MARKER}" if image else "")

        # Un seul appel à la fois par salon : l'historique reste cohérent.
        async with self.history.lock(message.channel.id):
            async with message.channel.typing():
                try:
                    answer = await self.claude.ask(self.history.get(message.channel.id), user_text, image or None)
                except ClaudeError as exc:
                    await self.safe_reply(message, exc.user_message)
                    return
                except Exception:
                    logger.exception("Erreur inattendue pendant le traitement d'un message")
                    await self.safe_reply(message, "Désolé, une erreur inattendue est survenue.")
                    return
            await self.history.add_exchange(message.channel.id, stored_text, answer)
        await self.send_answer(message, answer)

    async def handle_mention(self, message: discord.Message) -> None:
        """Hors du salon dédié : répond à une mention en lisant les messages récents."""
        lines: list[str] = []
        if self.settings.context_message_limit > 0:
            try:
                async for msg in message.channel.history(limit=self.settings.context_message_limit, before=message):
                    content = msg.clean_content.strip()
                    if msg.attachments:
                        content = (content + " [pièce jointe]").strip()
                    if msg.stickers:
                        content = (content + " [sticker]").strip()
                    if content:
                        lines.append(f"{msg.author.display_name} : {content}")
            except discord.Forbidden:
                logger.warning("Pas la permission de lire l'historique de #%s.", message.channel)
        lines.reverse()

        reply, image = await self.read_reply(message, want_image=True)

        bot_name = f"@{message.guild.me.display_name}" if message.guild else f"@{self.user.name}"
        text = message.clean_content.replace(bot_name, "").strip() or "(simple mention, sans texte)"
        channel_name = f"#{message.channel.name}" if hasattr(message.channel, "name") else "un message privé"

        parts = []
        if lines:
            parts.append(f"Contexte : derniers messages dans {channel_name}\n" + "\n".join(lines))
        reply_part = f" ({reply})" if reply else ""
        parts.append(f"{self.timestamp(message)}, {message.author.display_name} te mentionne{reply_part} et écrit : {text}")
        prompt = "\n\n".join(parts)

        async with message.channel.typing():
            try:
                answer = await self.claude.ask([], prompt, image)
            except ClaudeError as exc:
                await self.safe_reply(message, exc.user_message)
                return
            except Exception:
                logger.exception("Erreur inattendue pendant le traitement d'une mention")
                await self.safe_reply(message, "Désolé, une erreur inattendue est survenue.")
                return
        await self.send_answer(message, answer)

    # ------------------------------------------------------------- réactions

    async def reply_to_sticker(self, message: discord.Message) -> bool:
        """Répond à un sticker par un sticker aléatoire du serveur."""
        if not message.stickers:
            return False
        if message.guild is None:
            return True
        stickers = [s for s in message.guild.stickers if s.available]
        random.shuffle(stickers)
        for sticker in stickers:
            try:
                await message.channel.send(stickers=[sticker])
                logger.info("Sticker envoyé : %s", sticker.name)
                return True
            except discord.HTTPException as exc:
                logger.warning("Sticker %s inutilisable : %s", sticker.name, exc)
        await message.channel.send("Aucun sticker utilisable sur ce serveur.")
        return True

    async def reply_to_emoji(self, message: discord.Message) -> bool:
        """Répond à un message composé d'un seul emoji personnalisé par un emoji aléatoire du serveur."""
        if not CUSTOM_EMOJI_ONLY.match(message.content.strip()):
            return False
        emojis = [e for e in message.guild.emojis if e.available] if message.guild else []
        if not emojis:
            await message.channel.send("Aucun emoji personnalisé sur ce serveur.")
            return True
        try:
            await message.channel.send(str(random.choice(emojis)))
        except discord.HTTPException as exc:
            logger.warning("Impossible d'envoyer l'emoji : %s", exc)
        return True

    # ---------------------------------------------------------------- images

    async def read_image(self, message: discord.Message) -> ImageData | None | bool:
        """Vérifie les pièces jointes et télécharge l'image.

        Renvoie l'image, None s'il n'y en a pas, ou False si l'envoi est refusé
        (l'utilisateur est alors prévenu).
        """
        if not message.attachments:
            return None

        others = [a.filename for a in message.attachments if not looks_like_image(a)]
        if others:
            await self.safe_reply(message, f"Je ne peux lire que des images. Fichier(s) non pris en charge : {', '.join(others)}.")
            return False
        if len(message.attachments) > 1:
            await self.safe_reply(message, "Une seule image par message, s'il te plaît.")
            return False

        try:
            return await self.download_image(message.attachments[0])
        except ImageError as exc:
            await self.safe_reply(message, exc.user_message)
            return False

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

    # ------------------------------------------------------- message cité

    async def read_reply(self, message: discord.Message, want_image: bool) -> tuple[str | None, ImageData | None]:
        """Lit le message auquel l'utilisateur répond (bouton « Répondre » de Discord).

        Renvoie sa description pour Claude, ainsi que son image si want_image est vrai,
        ou (None, None) si le message ne répond à rien ou si le message cité est inaccessible.
        """
        ref = message.reference
        if ref is None or ref.message_id is None or ref.type != discord.MessageReferenceType.reply:
            return None, None

        cited = ref.resolved
        if cited is None:  # Discord ne l'a pas fourni (vieux message) : on le télécharge
            try:
                cited = await message.channel.fetch_message(ref.message_id)
            except discord.HTTPException as exc:
                logger.info("Message cité inaccessible : %s", exc)
                return None, None
        if not isinstance(cited, discord.Message):
            return None, None  # message cité supprimé

        content = cited.clean_content.strip()
        if len(content) > QUOTE_MAX_LENGTH:
            content = content[:QUOTE_MAX_LENGTH].rstrip() + "…"

        image = None
        images = [a for a in cited.attachments if looks_like_image(a)]
        if want_image and images:
            try:
                image = await self.download_image(images[0])
            except ImageError as exc:
                logger.warning("Image du message cité ignorée : %s", exc.user_message)

        parts = [f"« {content} »"] if content else []
        if image is not None:
            parts.append("[son image est jointe à ce message]")
        elif cited.attachments:
            parts.append("[pièce jointe]")
        if cited.stickers:
            parts.append("[sticker]")
        description = " ".join(parts) or "(message vide)"

        who = "à ton message" if cited.author == self.user else f"au message de {cited.author.display_name}"
        return f"en réponse {who} du {self.timestamp(cited)} : {description}", image

    # ------------------------------------------------------------- envoi

    def timestamp(self, message: discord.Message) -> str:
        return message.created_at.astimezone(self.settings.timezone).strftime("%d/%m/%Y %H:%M")

    async def send_answer(self, message: discord.Message, answer: str) -> None:
        chunks = split_message(answer)
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await message.reply(chunk, mention_author=False)
                else:
                    await message.channel.send(chunk)
            except discord.HTTPException as exc:
                # Le message d'origine a pu être supprimé entre-temps : on envoie sans répondre.
                if i == 0:
                    try:
                        await message.channel.send(chunk)
                        continue
                    except discord.HTTPException:
                        pass
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
    await bot.history.reset(interaction.channel_id)
    logger.info("Historique réinitialisé dans le salon %s par %s", interaction.channel_id, interaction.user)
    await interaction.response.send_message("L'historique de conversation a été réinitialisé.")
