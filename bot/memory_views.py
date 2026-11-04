"""Affichage et modification des fichiers de mémoire sur Discord : /memoire et /memoire-serveur.

Tout s'affiche dans des messages éphémères : seule la personne qui a lancé la commande les voit."""

from __future__ import annotations

import io
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import discord

from .members import MemberRegistry, display_name
from .memory import MAX_LINES, MAX_NICKNAME_LENGTH, MAX_NICKNAMES, NOT_SAVED, SERVER_FILE, MemoryStore
from .text_utils import split_message

logger = logging.getLogger("chatbot.memory")

# Longueur maximale d'une zone de texte dans une fenêtre Discord.
INPUT_MAX = 4000
# Choix d'un menu de Discord : 25 au plus, de 100 caractères. Menus au plus dans un message (5 rangées,
# dont une pour le bouton « Annuler ») : un fichier écrit avant la limite de MAX_LINES lignes peut en
# avoir plus de 25.
MENU_OPTIONS = 25
OPTION_LABEL_MAX = 100
MAX_MENUS = 4
# Texte maximal d'un embed (4096 pour Discord), et de tous les embeds d'un message (6000).
EMBED_MAX = 4000
EMBEDS_TOTAL_MAX = 5500
# Durée pendant laquelle les boutons répondent (ensuite, il suffit de relancer la commande).
VIEW_TIMEOUT = 600
# Durée pendant laquelle une fenêtre (« Ajouter des lignes », « Surnoms ») peut être validée : Discord la
# laisse ouverte sans limite, mais discord.py l'oublie ensuite et ignore sa validation.
MODAL_TIMEOUT = 3600
# Sous la fiche d'un membre, quand il la consulte lui-même.
ADMINS_NOTE = "Les administrateurs du serveur peuvent aussi voir ta fiche et en supprimer des lignes."
ERROR_MESSAGE = "Une erreur est survenue. Si ça se reproduit, préviens l'administrateur du bot."

Callback = Callable[[discord.Interaction], Awaitable[None]]


async def reply_error(interaction: discord.Interaction) -> None:
    """Prévient la personne qu'une commande, un bouton ou une fenêtre a échoué (le détail est dans le log),
    au lieu du seul « Échec de l'interaction » de Discord."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(ERROR_MESSAGE, ephemeral=True)
        else:
            await interaction.response.send_message(ERROR_MESSAGE, ephemeral=True)
    except discord.HTTPException as exc:
        logger.warning("Impossible de signaler l'erreur à %s : %s", interaction.user, exc)


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
    departed: bool = False  # membre parti du serveur : un administrateur désactive sa mémoire à sa place


class MemoryCard:
    """Ce que voit la personne qui a lancé la commande : le fichier et les boutons pour le modifier."""

    def __init__(self, store: MemoryStore, target: Target, owner_id: int, bot_name: str):
        self.store = store
        self.target = target
        self.owner_id = owner_id
        self.bot_name = bot_name
        self._view: OwnerView | None = None  # boutons affichés en ce moment

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
            if self.target.is_self:
                text = (f"La mémoire n'est pas activée pour toi : {self.bot_name} ne retient rien sur toi. "
                        f"Active-la pour que {self.bot_name} se souvienne de toi au fil des conversations.")
            else:
                text = (f"{self.target.name} n'a pas activé la mémoire : {self.bot_name} ne retient rien "
                        f"sur ce membre.")
            if content:  # fiche écrite avant que la mémoire soit à activer soi-même : ni lue ni complétée
                text += (f"\n\nUne ancienne fiche ({len(content)} caractères) est gardée : elle ne sert pas "
                         f"tant que la mémoire n'est pas activée.")
            return [discord.Embed(title=title, description=text, color=discord.Color.light_grey())]
        # Le membre doit savoir que sa fiche n'est pas vue par lui seul (voir /memoire membre:@…).
        note = ADMINS_NOTE if self.target.member_id is not None and self.target.is_self else ""
        if not content:
            embed = discord.Embed(title=title, description=f"{self.bot_name} n'a encore rien retenu {self.about()}.",
                                  color=discord.Color.blurple())
            if note:
                embed.set_footer(text=note)
            return [embed]
        size = len(content)
        content = self.store.with_names(content)  # membres cités par leur nom actuel (« <@222> » → « Alice »)
        if len(content) > EMBEDS_TOTAL_MAX:  # seulement si MEMORY_*_LIMIT dépasse ce que Discord affiche
            content = content[:EMBEDS_TOTAL_MAX] + "\n[…]"
        embeds = [discord.Embed(description=chunk, color=discord.Color.blurple())
                  for chunk in split_message(content, EMBED_MAX)]
        embeds[0].title = title
        footer = (f"{size}/{self.store.limit(self.target.file)} caractères, "
                  f"{len(self.store.lines(self.target.file))}/{MAX_LINES} lignes")
        embeds[-1].set_footer(text=f"{footer} · {note}" if note else footer)
        return embeds

    def view(self) -> discord.ui.View:
        view = OwnerView(self.owner_id)
        content = self.store.read(self.target.file)
        # Un fichier ne se modifie à la main que ligne par ligne (voir MemoryStore.delete_lines) : on en
        # supprime des lignes, seul l'administrateur ajoute des lignes au fichier général, et le membre
        # écrit ses surnoms. Mémoire pas activée : la fiche ne peut que perdre des lignes.
        if self.target.member_id is None:
            add_button(view, "Ajouter des lignes", "➕", discord.ButtonStyle.primary, self.ask_add)
        elif self.target.is_self:
            add_button(view, "Surnoms", "🏷️", discord.ButtonStyle.primary, self.ask_nicknames,
                       disabled=not self.memory_enabled)
        add_button(view, "Supprimer des lignes", "✂️", discord.ButtonStyle.secondary, self.choose_lines,
                   disabled=not content)
        add_button(view, "Tout effacer" if self.target.member_id is None else "Effacer la fiche", "🗑️",
                   discord.ButtonStyle.secondary, self.ask_erase, disabled=not content)
        # Seul le membre lui-même active ou désactive la mémoire. Parti du serveur, il ne peut plus lancer
        # /memoire : un administrateur peut alors la désactiver à sa place, pour tout effacer.
        if self.target.member_id is not None and (self.target.is_self or self.target.departed):
            if self.memory_enabled:
                add_button(view, "Désactiver la mémoire", "🚫", discord.ButtonStyle.danger, self.ask_disable)
            elif self.target.is_self:
                add_button(view, "Activer la mémoire", "✅", discord.ButtonStyle.success, self.ask_enable)
        return view

    async def send(self, interaction: discord.Interaction) -> None:
        await self.show(interaction, self.view(), self.embeds(), new=True)

    async def refresh(self, interaction: discord.Interaction, note: str | None = None) -> None:
        """Réaffiche le fichier à jour dans le même message, avec une courte phrase au-dessus."""
        await self.show(interaction, self.view(), self.embeds(), note)

    async def show(self, interaction: discord.Interaction, view: OwnerView, embeds: list[discord.Embed],
                   note: str | None = None, new: bool = False) -> None:
        """Affiche ces boutons dans un nouveau message éphémère (new), ou à la place des précédents."""
        if self._view is not None:
            self._view.stop()  # sans quoi son expiration griserait les nouveaux boutons
        self._view, view.interaction = view, interaction
        if new:
            await interaction.response.send_message(embeds=embeds, view=view, ephemeral=True)
        elif interaction.response.is_done():  # réponse différée (voir TextModal.on_submit)
            await interaction.edit_original_response(content=note, embeds=embeds, view=view)
        else:
            await interaction.response.edit_message(content=note, embeds=embeds, view=view)

    # -------------------------------------------------------------- actions

    async def choose_lines(self, interaction: discord.Interaction) -> None:
        """Remplace le fichier affiché par ses lignes numérotées et un menu pour cocher celles à supprimer."""
        original = self.store.read(self.target.file)  # les numéros valent pour ce contenu (voir delete_lines)
        lines = [self.store.with_names(line) for line in self.store.lines(self.target.file)]
        view = OwnerView(self.owner_id)
        for start in range(0, min(len(lines), MENU_OPTIONS * MAX_MENUS), MENU_OPTIONS):
            chunk = lines[start:start + MENU_OPTIONS]
            placeholder = "Lignes à supprimer"
            if len(lines) > MENU_OPTIONS:
                placeholder += f" ({start + 1} à {start + len(chunk)})"
            menu = discord.ui.Select(
                placeholder=placeholder, min_values=1, max_values=len(chunk),
                options=[discord.SelectOption(label=shorten(f"{start + i + 1}. {line}"), value=str(start + i))
                         for i, line in enumerate(chunk)])
            menu.callback = self._deleter(menu, original)
            view.add_item(menu)
        add_button(view, "Annuler", "↩️", discord.ButtonStyle.secondary, self.refresh)
        text = "\n".join(f"**{i + 1}.** {line}" for i, line in enumerate(lines))
        if len(lines) > MENU_OPTIONS * MAX_MENUS:
            text += (f"\n\nSeules les {MENU_OPTIONS * MAX_MENUS} premières lignes peuvent être choisies : "
                     f"supprimes-en d'abord, les suivantes apparaîtront.")
        embeds = [discord.Embed(description=chunk, color=discord.Color.orange())
                  for chunk in split_message(text[:EMBEDS_TOTAL_MAX], EMBED_MAX)]
        embeds[0].title = "Coche les lignes à supprimer (c'est définitif)"
        await self.show(interaction, view, embeds)

    def _deleter(self, menu: discord.ui.Select, original: str) -> Callback:
        """Action du menu : supprime les lignes cochées, puis réaffiche le fichier."""
        async def delete(interaction: discord.Interaction) -> None:
            indexes = [int(value) for value in menu.values]
            error = await self.store.delete_lines(self.target.file, indexes, by=str(interaction.user),
                                                  expected=original)
            plural = "s" if len(indexes) > 1 else ""
            await self.refresh(interaction, error or f"{len(indexes)} ligne{plural} supprimée{plural}.")
        return delete

    async def ask_add(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(TextModal(
            self, "Ajouter au fichier général", "Lignes à ajouter", self.store.add_lines,
            placeholder=f"Une information par ligne ({MAX_LINES} lignes au plus dans le fichier).",
            max_length=min(INPUT_MAX, self.store.limit(self.target.file)), done="C'est ajouté."))

    async def ask_nicknames(self, interaction: discord.Interaction) -> None:
        async def save(_file: str, text: str, by: str) -> str | None:
            return await self.store.set_nicknames(self.target.member_id, text, by=by)
        current = ", ".join(self.store.nicknames(self.target.member_id))
        await interaction.response.send_modal(TextModal(
            self, "Tes surnoms", "Surnoms, séparés par des virgules", save, default=current,
            placeholder="Vide : aucun surnom.", max_length=MAX_NICKNAMES * (MAX_NICKNAME_LENGTH + 2),
            paragraph=False, done="C'est enregistré."))

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
        error = await self.store.erase(self.target.file, by=str(interaction.user))
        await self.refresh(interaction, error or "C'est effacé.")

    async def ask_disable(self, interaction: discord.Interaction) -> None:
        # Le fichier général n'est pas touché (voir MemoryStore.forget) : la question ne promet pas plus.
        if self.target.departed:
            question = (f"{self.target.name} a quitté le serveur. Désactiver sa mémoire efface définitivement sa "
                        f"fiche et retire son nom des sources (« d'après … ») dans les fiches des autres. Le "
                        f"fichier général de {self.bot_name} n'est pas modifié : vérifie avec /memoire-serveur "
                        f"qu'il ne parle plus de ce membre.")
            await confirm(interaction, self, question, self.disable)
            return
        question = (f"Si tu désactives la mémoire, ta fiche sera définitivement effacée, et ton nom sera retiré "
                    f"des sources (« d'après … ») dans les fiches des autres. {self.bot_name} ne notera plus "
                    f"rien sur toi, mais ce que tu as dit récemment reste dans la conversation. Le fichier "
                    f"général du serveur n'est pas modifié : pour vérifier qu'il ne parle pas de toi, "
                    f"rapproche-toi d'un administrateur du serveur. Tu pourras réactiver la mémoire plus tard "
                    f"avec /memoire.")
        await confirm(interaction, self, question, self.disable)

    async def disable(self, interaction: discord.Interaction) -> None:
        error = await self.store.forget(self.target.member_id, by=str(interaction.user))
        if self.target.departed:
            done = (f"Mémoire désactivée pour {self.target.name} : sa fiche a été effacée, et son nom retiré des "
                    f"sources. Vérifie avec /memoire-serveur que le fichier général ne parle plus de ce membre.")
        else:
            done = ("Mémoire désactivée : ta fiche a été effacée, et ton nom retiré des sources. Pour vérifier "
                    "que le fichier général du serveur ne parle pas de toi, rapproche-toi d'un administrateur "
                    "du serveur.")
        await self.refresh(interaction, error or done)

    async def ask_enable(self, interaction: discord.Interaction) -> None:
        # Seule présentation de la mémoire au membre : tout ce qu'il doit savoir avant d'accepter.
        question = (f"Si tu actives la mémoire, {self.bot_name} retiendra des choses sur toi au fil des "
                    f"conversations (tes goûts, tes projets…), y compris ce que d'autres membres lui racontent "
                    f"sur toi, avec leur nom comme source (« d'après … »). De même, ce que tu lui racontes sur "
                    f"un autre membre qui a activé la mémoire peut être noté dans sa fiche avec ton nom, et il "
                    f"la voit. Ce qui est retenu sert dans la conversation, mais ta fiche n'est jamais récitée "
                    f"aux autres ; à part toi, seuls les administrateurs du serveur peuvent la voir et en "
                    f"supprimer des lignes. Avec /memoire, tu pourras la voir, en supprimer des lignes ou désactiver la mémoire.")
        await confirm(interaction, self, question, self.enable)

    async def enable(self, interaction: discord.Interaction) -> None:
        saved = await self.store.members.set_memory(self.target.member_id, True)
        await self.refresh(interaction, "Mémoire activée." if saved else NOT_SAVED.format(setting="l'activation"))


class OwnerView(discord.ui.View):
    """Boutons réservés à la personne qui a lancé la commande."""

    def __init__(self, owner_id: int):
        super().__init__(timeout=VIEW_TIMEOUT)
        self.owner_id = owner_id
        self.interaction: discord.Interaction | None = None  # celle qui a affiché ces boutons (voir show)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id

    async def on_timeout(self) -> None:
        """Boutons grisés : ils ne répondent plus. Le message éphémère reste modifiable 15 minutes après
        l'interaction qui l'a affiché, plus que VIEW_TIMEOUT."""
        if self.interaction is None:
            return
        for item in self.children:
            item.disabled = True
        try:
            await self.interaction.edit_original_response(
                content="Ces boutons ont expiré : relance la commande pour continuer.", view=self)
        except discord.HTTPException:
            pass  # message fermé par la personne, ou trop ancien

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item) -> None:
        logger.exception("Erreur sur le bouton « %s »", getattr(item, "label", "?"), exc_info=error)
        await reply_error(interaction)


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
    await card.show(interaction, view, [embed])


class TextModal(discord.ui.Modal):
    """Fenêtre d'une seule zone de texte (« Ajouter des lignes », « Surnoms ») : save(fichier, texte, by)
    renvoie un message d'erreur, ou None."""

    def __init__(self, card: MemoryCard, title: str, label: str,
                 save: Callable[[str, str, str], Awaitable[str | None]], max_length: int, done: str,
                 default: str = "", placeholder: str | None = None, paragraph: bool = True):
        super().__init__(title=title[:45], timeout=MODAL_TIMEOUT)
        self.card, self.save, self.done = card, save, done
        self.input = discord.ui.TextInput(
            label=label[:45], style=discord.TextStyle.paragraph if paragraph else discord.TextStyle.short,
            required=False, max_length=max_length, default=default or None, placeholder=placeholder)
        self.add_item(self.input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        text = self.input.value.strip()
        # Discord laisse 3 secondes pour répondre, et l'écriture sur le disque peut prendre plus longtemps
        # (fsync sur un volume réseau) : sans réponse, la fenêtre resterait ouverte.
        await interaction.response.defer()
        error = await self.save(self.card.target.file, text, str(interaction.user))
        if error:
            # Le texte saisi est renvoyé en fichier joint (trop long pour un message) : rien n'est perdu.
            file = discord.File(io.BytesIO(text.encode("utf-8")), filename="texte_non_enregistre.txt")
            await interaction.followup.send(f"{error} Ton texte est joint, pour le recopier.",
                                            file=file, ephemeral=True)
            return
        await self.card.refresh(interaction, self.done)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        logger.exception("Erreur pendant la modification de %s", self.card.target.file, exc_info=error)
        await reply_error(interaction)


def shorten(text: str) -> str:
    """Texte d'un choix de menu, coupé à la limite de Discord."""
    return text if len(text) <= OPTION_LABEL_MAX else text[:OPTION_LABEL_MAX - 1] + "…"


def member_target(store: MemoryStore, member: discord.User | discord.Member, owner_id: int) -> Target:
    entry = store.members.get(member.id)
    name = display_name(entry) if entry else member.display_name
    # Hors du serveur, Discord donne un discord.User au lieu d'un discord.Member.
    is_self = member.id == owner_id
    departed = not is_self and not isinstance(member, discord.Member)
    return Target(store.member_file(member.id), member.id, name, is_self=is_self, departed=departed)


def server_target() -> Target:
    return Target(SERVER_FILE, None)
