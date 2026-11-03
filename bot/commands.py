"""Commandes slash : /reset, /skip, /memoire et /memoire-serveur (enregistrées par ChatBot.setup_hook)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands

from .memory_views import MemoryCard, member_target, see_user, server_target

if TYPE_CHECKING:
    from .discord_bot import ChatBot

logger = logging.getLogger("chatbot.discord")  # même nom qu'avant : le journal ne change pas

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
