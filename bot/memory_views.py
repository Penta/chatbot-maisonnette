"""Affichage et modification des fichiers de mémoire sur Discord : /memoire et /memoire-serveur.

Tout s'affiche dans des messages éphémères : seule la personne qui a lancé la commande les voit."""

from __future__ import annotations

import io
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import discord

from .members import MemberRegistry, display_name
from .memory import NOT_SAVED, SERVER_FILE, MemoryStore
from .text_utils import split_message

logger = logging.getLogger("chatbot.memory")

# Longueur maximale d'une zone de texte dans une fenêtre Discord, et nombre maximal de zones.
INPUT_MAX = 4000
MAX_INPUTS = 5
# Texte maximal d'un embed (4096 pour Discord), et de tous les embeds d'un message (6000).
EMBED_MAX = 4000
EMBEDS_TOTAL_MAX = 5500
# Durée pendant laquelle les boutons répondent (ensuite, il suffit de relancer la commande).
VIEW_TIMEOUT = 600
# Sous la fiche d'un membre, quand il la consulte lui-même.
ADMINS_NOTE = "Les administrateurs du serveur peuvent aussi voir et corriger ta fiche."

Callback = Callable[[discord.Interaction], Awaitable[None]]


async def see_user(registry: MemberRegistry, user: discord.User | discord.Member) -> None:
    """Ajoute un utilisateur Discord au registre, ou met ses noms à jour."""
    # En message privé, l'auteur n'est pas un membre du serveur : son pseudo n'est pas connu.
    in_guild = isinstance(user, discord.Member)
    await registry.see(user.id, user.name, user.global_name, user.nick if in_guild else None, in_guild=in_guild)


@dataclass
class Target:
    """Fichier affiché : la fiche d'un membre, ou le fichier général (member_id=None)."""
    file: str
    member_id: int | None
    name: str = ""        # nom du membre
    is_self: bool = False  # fiche de la personne qui a lancé la commande


class MemoryCard:
    """Ce que voit la personne qui a lancé la commande : le fichier et les boutons pour le modifier."""

    def __init__(self, store: MemoryStore, target: Target, owner_id: int, bot_name: str):
        self.store = store
        self.target = target
        self.owner_id = owner_id
        self.bot_name = bot_name

    # ---------------------------------------------------------------- état

    @property
    def memory_enabled(self) -> bool:
        return self.target.member_id is None or self.store.members.memory_enabled(self.target.member_id)

    def about(self) -> str:
        """« sur toi », « sur Penta » ou « sur le serveur »."""
        if self.target.member_id is None:
            return "sur le serveur"
        return "sur toi" if self.target.is_self else f"sur {self.target.name}"

    # ----------------------------------------------------------- affichage

    def embeds(self) -> list[discord.Embed]:
        if self.target.member_id is None:
            title = "Fichier général du serveur"
        else:
            title = "Ta fiche mémoire" if self.target.is_self else f"Fiche mémoire de {self.target.name}"
        content = self.store.read(self.target.file)
        if not self.memory_enabled:
            who = "Tu as" if self.target.is_self else f"{self.target.name} a"
            text = (f"{who} désactivé la mémoire : {self.bot_name} ne retient rien "
                    f"{self.about()} et ne voit aucune fiche dans ses messages.")
            return [discord.Embed(title=title, description=text, color=discord.Color.light_grey())]
        # Le membre doit savoir que sa fiche n'est pas vue par lui seul (voir /memoire membre:@…).
        admins = ADMINS_NOTE if self.target.member_id is not None and self.target.is_self else ""
        if not content:
            embed = discord.Embed(title=title, description=f"{self.bot_name} n'a encore rien retenu {self.about()}.",
                                  color=discord.Color.blurple())
            if admins:
                embed.set_footer(text=admins)
            return [embed]
        size = len(content)
        if size > EMBEDS_TOTAL_MAX:  # seulement si MEMORY_*_LIMIT dépasse ce que Discord affiche
            content = content[:EMBEDS_TOTAL_MAX] + "\n[…] (la suite est visible dans « Modifier »)"
        embeds = [discord.Embed(description=chunk, color=discord.Color.blurple())
                  for chunk in split_message(content, EMBED_MAX)]
        embeds[0].title = title
        footer = f"{size}/{self.store.limit(self.target.file)} caractères"
        embeds[-1].set_footer(text=f"{footer} · {admins}" if admins else footer)
        return embeds

    def view(self) -> discord.ui.View:
        view = OwnerView(self.owner_id)
        content = self.store.read(self.target.file)
        add_button(view, "Modifier", "✏️", discord.ButtonStyle.primary, self.edit, disabled=not self.memory_enabled)
        add_button(view, "Tout effacer" if self.target.member_id is None else "Effacer la fiche", "🗑️",
                   discord.ButtonStyle.secondary, self.ask_erase, disabled=not content or not self.memory_enabled)
        # Seul le membre lui-même choisit d'activer ou non la mémoire.
        if self.target.member_id is not None and self.target.is_self:
            if self.memory_enabled:
                add_button(view, "Désactiver la mémoire", "🚫", discord.ButtonStyle.danger, self.ask_disable)
            else:
                add_button(view, "Réactiver la mémoire", "✅", discord.ButtonStyle.success, self.enable)
        return view

    async def send(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(embeds=self.embeds(), view=self.view(), ephemeral=True)

    async def refresh(self, interaction: discord.Interaction, note: str | None = None) -> None:
        """Réaffiche le fichier à jour dans le même message, avec une courte phrase au-dessus."""
        await interaction.response.edit_message(content=note, embeds=self.embeds(), view=self.view())

    # -------------------------------------------------------------- actions

    async def edit(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(EditModal(self))

    async def save(self, interaction: discord.Interaction, text: str, original: str | None = None) -> None:
        """Enregistre le texte de la fenêtre « Modifier » (original : le fichier à son ouverture)."""
        error = await self.store.overwrite(self.target.file, text, by=str(interaction.user), expected=original)
        if error:
            if not text.strip():
                await interaction.response.send_message(error, ephemeral=True)
                return
            # Le texte saisi est renvoyé en fichier joint (trop long pour un message) : rien n'est perdu.
            file = discord.File(io.BytesIO(text.encode("utf-8")), filename="texte_non_enregistre.txt")
            await interaction.response.send_message(f"{error} Ton texte est joint, pour le recopier.",
                                                    file=file, ephemeral=True)
            return
        await self.refresh(interaction, "C'est enregistré." if text.strip() else "C'est effacé.")

    async def ask_erase(self, interaction: discord.Interaction) -> None:
        # Seule la fiche est effacée (pas le fichier général, ni la conversation) : la question ne promet pas plus.
        if self.target.member_id is None:
            question = "Effacer tout le fichier général du serveur ? C'est définitif."
        else:
            whose = "ta fiche" if self.target.is_self else f"la fiche de {self.target.name}"
            said = "tu as dit" if self.target.is_self else f"{self.target.name} a dit"
            question = (f"Effacer {whose} ? C'est définitif. Seule la fiche est effacée : ce que {said} "
                        f"récemment reste dans la conversation, et {self.bot_name} peut encore s'en souvenir.")
        await confirm(interaction, self, question, self.erase)

    async def erase(self, interaction: discord.Interaction) -> None:
        error = await self.store.overwrite(self.target.file, "", by=str(interaction.user))
        await self.refresh(interaction, error or "C'est effacé.")

    async def ask_disable(self, interaction: discord.Interaction) -> None:
        question = (f"Si tu désactives la mémoire, ta fiche et les lignes du fichier général de "
                    f"{self.bot_name} qui te citent seront définitivement effacées, et ton nom sera retiré "
                    f"des sources (« d'après … ») dans les fiches des autres. {self.bot_name} ne notera plus "
                    f"rien sur toi, mais ce que tu as dit récemment reste dans la conversation. "
                    f"Tu pourras la réactiver plus tard avec /memoire.")
        await confirm(interaction, self, question, self.disable)

    async def disable(self, interaction: discord.Interaction) -> None:
        error = await self.store.forget(self.target.member_id, by=str(interaction.user))
        await self.refresh(interaction, error or "Mémoire désactivée : ta fiche et ce qui te citait dans le "
                                                 "fichier général ont été effacés, et ton nom retiré des sources.")

    async def enable(self, interaction: discord.Interaction) -> None:
        saved = await self.store.members.set_memory(self.target.member_id, True)
        await self.refresh(interaction, "Mémoire réactivée." if saved else NOT_SAVED.format(setting="la réactivation"))


class OwnerView(discord.ui.View):
    """Boutons réservés à la personne qui a lancé la commande."""

    def __init__(self, owner_id: int):
        super().__init__(timeout=VIEW_TIMEOUT)
        self.owner_id = owner_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id


def add_button(view: discord.ui.View, label: str, emoji: str, style: discord.ButtonStyle,
               callback: Callback, disabled: bool = False) -> None:
    button = discord.ui.Button(label=label, emoji=emoji, style=style, disabled=disabled)
    button.callback = callback
    view.add_item(button)


async def confirm(interaction: discord.Interaction, card: MemoryCard, question: str, action: Callback) -> None:
    """Remplace le fichier affiché par une question, avec « Confirmer » et « Annuler »."""
    view = OwnerView(card.owner_id)
    add_button(view, "Confirmer", "⚠️", discord.ButtonStyle.danger, action)
    add_button(view, "Annuler", "↩️", discord.ButtonStyle.secondary, card.refresh)
    embed = discord.Embed(description=question, color=discord.Color.orange())
    await interaction.response.edit_message(content=None, embeds=[embed], view=view)


class EditModal(discord.ui.Modal):
    """Fenêtre « Modifier » : le fichier, déjà rempli, dans une ou plusieurs zones de texte
    (une zone fait 4000 caractères au plus ; le fichier général peut en demander deux)."""

    def __init__(self, card: MemoryCard):
        title = "Modifier ta fiche" if card.target.is_self else (
            "Modifier le fichier général" if card.target.member_id is None else f"Fiche de {card.target.name}")
        super().__init__(title=title[:45], timeout=VIEW_TIMEOUT)
        self.card = card
        limit = card.store.limit(card.target.file)
        count = min(MAX_INPUTS, -(-limit // INPUT_MAX))
        self.original = card.store.read(card.target.file)  # comparé à l'enregistrement (voir overwrite)
        parts = split_for_inputs(self.original, count)
        self.inputs = []
        for i in range(count):
            label = "Contenu" if count == 1 else f"Contenu ({i + 1}/{count})"
            text_input = discord.ui.TextInput(
                label=label, style=discord.TextStyle.paragraph, required=False,
                max_length=min(INPUT_MAX, limit), default=parts[i] or None,
                placeholder="Une information par ligne. Tout vider efface le fichier." if i == 0 else None,
            )
            self.inputs.append(text_input)
            self.add_item(text_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        text = "\n".join(text_input.value.strip() for text_input in self.inputs if text_input.value.strip())
        await self.card.save(interaction, text, self.original)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        logger.exception("Erreur pendant la modification de %s", self.card.target.file, exc_info=error)
        if not interaction.response.is_done():
            await interaction.response.send_message("La modification a échoué, regarde les logs.", ephemeral=True)


def split_for_inputs(text: str, count: int) -> list[str]:
    """Répartit un texte dans count zones de texte, de préférence entre les lignes."""
    parts = split_message(text, INPUT_MAX)
    if len(parts) > count:  # lignes trop mal réparties : découpage exact
        parts = [text[i:i + INPUT_MAX] for i in range(0, len(text), INPUT_MAX)]
    return (parts + [""] * count)[:count]


def member_target(store: MemoryStore, member: discord.User | discord.Member, owner_id: int) -> Target:
    entry = store.members.get(member.id)
    name = display_name(entry) if entry else member.display_name
    return Target(store.member_file(member.id), member.id, name, is_self=member.id == owner_id)


def server_target() -> Target:
    return Target(SERVER_FILE, None)
