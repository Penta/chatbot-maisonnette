"""Affichage et modification des fichiers de mémoire sur Discord : /memoire et /memoire-serveur.

Tout s'affiche dans des messages éphémères : seule la personne qui a lancé la commande les voit. Ces messages
utilisent la mise en page de Discord par blocs (textes, rangées de boutons) : chaque ligne s'y lit en entier,
alors qu'un menu déroulant la couperait à 100 caractères."""

from __future__ import annotations

import io
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import discord

from .members import MemberRegistry, display_name
from .memory import (MAX_LINES, MAX_NICKNAME_LENGTH, MAX_NICKNAMES, NICKNAMES_LINE, NOT_SAVED, SERVER_FILE,
                     MemoryStore)

logger = logging.getLogger("chatbot.memory")

# Longueur maximale d'une zone de texte dans une fenêtre Discord.
INPUT_MAX = 4000
# Limites de Discord pour un message mis en page par blocs : 40 éléments au plus (textes, rangées, boutons,
# chacun compte) et 4000 caractères de texte en tout.
TEXT_MAX = 4000
MAX_ITEMS = 40
# Texte du fichier sur une page : le reste du message (titre, bas de page, phrase au-dessus, avis
# d'expiration) tient dans la marge. Un fichier plus long s'affiche en plusieurs pages.
PAGE_TEXT_MAX = 3300
# « Supprimer des lignes » : un bouton numéroté par ligne, en rangées de 5. 25 boutons par page (5 rangées,
# plus une pour les autres boutons) restent sous les 40 éléments ; une fiche de MAX_LINES lignes tient ainsi
# sur une seule page.
NUMBERS_PER_ROW = 5
NUMBERS_PER_PAGE = 25
# Durée pendant laquelle les boutons répondent (ensuite, il suffit de relancer la commande).
VIEW_TIMEOUT = 600
# Durée pendant laquelle une fenêtre (« Ajouter des lignes », « Surnoms ») peut être validée : Discord la
# laisse ouverte sans limite, mais discord.py l'oublie ensuite et ignore sa validation.
MODAL_TIMEOUT = 3600
# ANSWER_FIRST : Discord laisse 3 secondes pour répondre à un clic, et une écriture peut prendre plus longtemps
# (verrou de la mémoire, fsync sur un volume réseau). Toute action qui écrit commence donc par interaction.response
# .defer() : sinon « Échec de l'interaction » s'affiche alors que c'est fait, et la personne croit que rien n'a changé.
# Sous la fiche d'un membre, quand il la consulte lui-même.
ADMINS_NOTE = "Les administrateurs du serveur peuvent aussi voir ta fiche, en supprimer des lignes ou l'effacer."
ERROR_MESSAGE = "Une erreur est survenue. Si ça se reproduit, préviens l'administrateur du bot."
EXPIRED = "Ces boutons ont expiré : relance la commande pour continuer."
CHOOSE_HELP = ("Clique sur les numéros des lignes à supprimer : elles apparaissent barrées. Rien n'est "
               "supprimé avant le bouton « Supprimer », et la suppression est définitive.")
EDIT_HELP = "Clique sur le numéro de la ligne à modifier : son texte s'ouvre dans une fenêtre, prêt à corriger."

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


@dataclass
class Choice:
    """Écran « Supprimer des lignes » (ou « Modifier une ligne », edit) en cours : lignes et numéros cochés."""
    original: str          # contenu du fichier à l'ouverture : les numéros valent pour lui (voir delete_lines)
    raw: list[str]         # lignes telles qu'enregistrées (membres cités par « <@id> »)
    lines: list[str]       # lignes telles qu'affichées (membres cités par leur nom)
    edit: bool = False
    chosen: set[int] = field(default_factory=set)  # index des lignes cochées
    page: int = 0


class MemoryCard:
    """Ce que voit la personne qui a lancé la commande : le fichier et les boutons pour le modifier."""

    def __init__(self, store: MemoryStore, target: Target, owner_id: int, bot_name: str):
        self.store = store
        self.target = target
        self.owner_id = owner_id
        self.bot_name = bot_name
        self.page = 0  # page affichée d'un fichier trop long pour un seul message
        self._choice: Choice | None = None
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

    def view(self, note: str | None = None) -> OwnerView:
        """Écran principal : le fichier, ses boutons, et une courte phrase au-dessus (note) après une action."""
        view = OwnerView(self.owner_id)
        if note:
            view.add_item(discord.ui.TextDisplay(note))
        if self.target.member_id is None:
            title = "Fichier général du serveur"
        else:
            title = "Ta fiche mémoire" if self.target.is_self else f"Fiche mémoire de {self.target.name}"
        content = self.store.read(self.target.file)
        # Le membre doit savoir que sa fiche n'est pas vue par lui seul (voir /memoire membre:@…).
        footer = [ADMINS_NOTE] if self.target.member_id is not None and self.target.is_self else []
        color, pages = discord.Color.blurple(), 1
        if not self.memory_enabled:
            color, footer = discord.Color.light_grey(), []
            if self.target.is_self:
                text = (f"La mémoire n'est pas activée pour toi : {self.bot_name} ne retient rien sur toi. "
                        f"Active-la pour que {self.bot_name} se souvienne de toi au fil des conversations.")
            else:
                text = (f"{self.target.name} n'a pas activé la mémoire : {self.bot_name} ne retient rien "
                        f"sur ce membre.")
        elif not content:
            text = f"{self.bot_name} n'a encore rien retenu {self.about()}."
        else:
            # Membres cités par leur nom actuel (« <@222> » → « Alice »).
            lines = [clip(line) for line in self.store.with_names(content).split("\n")]
            bounds = paginate([len(line) for line in lines], PAGE_TEXT_MAX)
            pages, self.page = len(bounds), min(self.page, len(bounds) - 1)
            start, end = bounds[self.page]
            text = "\n".join(lines[start:end])
            footer.insert(0, f"{len(content)}/{self.store.limit(self.target.file)} caractères, "
                             f"{len(self.store.lines(self.target.file))}/{MAX_LINES} lignes")
        body = f"### {title}\n{text}" + "".join(f"\n-# {line}" for line in footer)
        view.add_item(discord.ui.Container(discord.ui.TextDisplay(body), accent_colour=color))
        for row in self.buttons(content):
            view.add_item(discord.ui.ActionRow(*row))
        if pages > 1:
            view.add_item(discord.ui.ActionRow(*page_buttons(self.page, pages, self.turn_page)))
        return view

    def buttons(self, content: str) -> list[list[discord.ui.Button]]:
        """Rangées de boutons de l'écran principal : écrire, puis retirer."""
        write, buttons = [], []
        # Un fichier ne se modifie à la main que ligne par ligne (voir MemoryStore.delete_lines) : chacun en
        # supprime des lignes ; l'administrateur ajoute ou modifie des lignes du fichier général, et le membre
        # de sa propre fiche, dont il écrit aussi les surnoms. Mémoire pas activée : il n'y a pas de fiche.
        if self.target.member_id is None or self.target.is_self:
            off = not self.memory_enabled
            write.append(button("Ajouter des lignes", self.ask_add, discord.ButtonStyle.primary, "➕", disabled=off))
            write.append(button("Modifier une ligne", self.choose_line_to_edit, discord.ButtonStyle.primary, "✏️",
                                disabled=off or not content))
            if self.target.is_self:
                write.append(button("Surnoms", self.ask_nicknames, discord.ButtonStyle.primary, "🏷️", disabled=off))
        buttons.append(button("Supprimer des lignes", self.choose_lines, emoji="✂️", disabled=not content))
        buttons.append(button("Tout effacer" if self.target.member_id is None else "Effacer la fiche",
                              self.ask_erase, emoji="🗑️", disabled=not content))
        # Seul le membre lui-même active ou désactive la mémoire. Parti du serveur, il ne peut plus lancer
        # /memoire : un administrateur peut alors la désactiver à sa place, pour tout effacer.
        if self.target.member_id is not None and (self.target.is_self or self.target.departed):
            if self.memory_enabled:
                buttons.append(button("Désactiver la mémoire", self.ask_disable, discord.ButtonStyle.danger, "🚫"))
            elif self.target.is_self:
                buttons.append(button("Activer la mémoire", self.ask_enable, discord.ButtonStyle.success, "✅"))
        return [row for row in (write, buttons) if row]

    async def turn_page(self, interaction: discord.Interaction, page: int) -> None:
        self.page = page
        await self.refresh(interaction)

    async def send(self, interaction: discord.Interaction) -> None:
        await self.show(interaction, self.view(), new=True)

    async def refresh(self, interaction: discord.Interaction, note: str | None = None) -> None:
        """Réaffiche le fichier à jour dans le même message, avec une courte phrase au-dessus."""
        await self.show(interaction, self.view(note))

    async def show(self, interaction: discord.Interaction, view: OwnerView, new: bool = False) -> None:
        """Affiche cet écran dans un nouveau message éphémère (new), ou à la place du précédent."""
        if self._view is not None:
            self._view.stop()  # sans quoi son expiration griserait les nouveaux boutons
        self._view, view.interaction = view, interaction
        if new and interaction.response.is_done():  # réponse différée (voir memory_command)
            await interaction.followup.send(view=view, ephemeral=True)
        elif new:
            await interaction.response.send_message(view=view, ephemeral=True)
        elif interaction.response.is_done():  # réponse différée, avant une écriture (voir ANSWER_FIRST)
            await interaction.edit_original_response(view=view)
        else:
            await interaction.response.edit_message(view=view)

    # -------------------------------------------------------------- actions

    async def choose_lines(self, interaction: discord.Interaction) -> None:
        """Remplace le fichier affiché par ses lignes numérotées, avec un bouton par numéro pour les cocher."""
        await self.open_choice(interaction, edit=False)

    async def choose_line_to_edit(self, interaction: discord.Interaction) -> None:
        """Comme choose_lines, mais un numéro ouvre la fenêtre pour modifier sa ligne."""
        await self.open_choice(interaction, edit=True)

    async def open_choice(self, interaction: discord.Interaction, edit: bool) -> None:
        # Bouton affiché avant un changement : fiche effacée entre-temps (autre message /memoire, désactivation).
        original = self.store.read(self.target.file)
        if not original:
            await self.refresh(interaction, f"Le fichier est vide : rien à {'modifier' if edit else 'supprimer'}.")
            return
        raw = self.store.lines(self.target.file)
        self._choice = Choice(original, raw, [self.store.with_names(line) for line in raw], edit=edit)
        await self.show(interaction, self.choice_view())

    def choice_view(self) -> OwnerView:
        """Écran « Supprimer des lignes » (les lignes en entier, les cochées barrées, et leurs boutons) ou
        « Modifier une ligne » (les mêmes, sans rien à cocher)."""
        choice = self._choice
        # Pages calculées comme si tout était coché (barré) : elles ne bougent pas quand on coche.
        bounds = paginate([len(numbered(i, line, True)) for i, line in enumerate(choice.lines)], PAGE_TEXT_MAX,
                          NUMBERS_PER_PAGE)
        choice.page = min(choice.page, len(bounds) - 1)
        start, end = bounds[choice.page]
        text = "\n".join(numbered(i, choice.lines[i], i in choice.chosen) for i in range(start, end))
        heading, help_text = ("Modifier une ligne", EDIT_HELP) if choice.edit else ("Lignes à supprimer", CHOOSE_HELP)
        body = f"### {heading}\n{help_text}\n\n{text}"
        if len(bounds) > 1:
            body += f"\n-# Lignes {start + 1} à {end} sur {len(choice.lines)}"
        container = discord.ui.Container(discord.ui.TextDisplay(body), accent_colour=discord.Color.orange())
        for row in range(start, end, NUMBERS_PER_ROW):
            container.add_item(discord.ui.ActionRow(
                *(self.number_button(i) for i in range(row, min(row + NUMBERS_PER_ROW, end)))))
        view = OwnerView(self.owner_id)
        view.add_item(container)
        count = len(choice.chosen)
        plural = "s" if count > 1 else ""
        buttons = page_buttons(choice.page, len(bounds), self.turn_choice_page, label=False) if len(bounds) > 1 else []
        if not choice.edit:
            buttons.append(button(f"Supprimer {count} ligne{plural}" if count else "Supprimer", self.delete_chosen,
                                  discord.ButtonStyle.danger, "✂️", disabled=not count))
        buttons.append(button("Annuler", self.refresh, emoji="↩️"))
        view.add_item(discord.ui.ActionRow(*buttons))
        return view

    def number_button(self, index: int) -> discord.ui.Button:
        """Bouton d'une ligne : un clic la coche (rouge, barrée dans la liste), un autre la décoche ; pour
        « Modifier une ligne », un clic ouvre la fenêtre de la ligne."""
        chosen = index in self._choice.chosen

        async def toggle(interaction: discord.Interaction) -> None:
            self._choice.chosen.symmetric_difference_update({index})
            await self.show(interaction, self.choice_view())

        async def edit(interaction: discord.Interaction) -> None:
            await self.ask_edit(interaction, index)
        return button(str(index + 1), edit if self._choice.edit else toggle,
                      discord.ButtonStyle.danger if chosen else discord.ButtonStyle.secondary)

    async def ask_edit(self, interaction: discord.Interaction, index: int) -> None:
        """Fenêtre pour modifier une ligne, son texte déjà rempli. La ligne des surnoms a sa propre fenêtre."""
        choice = self._choice
        line = choice.raw[index]
        if self.target.member_id is not None and NICKNAMES_LINE.match(line.strip()):
            await self.ask_nicknames(interaction)
            return

        async def save(file: str, text: str, by: str) -> str | None:
            return await self.store.edit_line(file, index, text, by=by, expected=choice.original)
        # Membres cités par « @alice », reconverti à l'enregistrement : la source reste liée au membre.
        await interaction.response.send_modal(TextModal(
            self, f"Modifier la ligne {index + 1}", "Texte de la ligne", save,
            default=self.store.with_usernames(line), placeholder="Vide : la ligne est supprimée.",
            max_length=min(INPUT_MAX, self.store.limit(self.target.file)), done="C'est enregistré."))

    async def turn_choice_page(self, interaction: discord.Interaction, page: int) -> None:
        self._choice.page = page
        await self.show(interaction, self.choice_view())

    async def delete_chosen(self, interaction: discord.Interaction) -> None:
        """Supprime les lignes cochées, puis réaffiche le fichier."""
        await interaction.response.defer()  # voir ANSWER_FIRST
        indexes = sorted(self._choice.chosen)
        error = await self.store.delete_lines(self.target.file, indexes, by=str(interaction.user),
                                              expected=self._choice.original)
        plural = "s" if len(indexes) > 1 else ""
        await self.refresh(interaction, error or f"{len(indexes)} ligne{plural} supprimée{plural}.")

    async def ask_add(self, interaction: discord.Interaction) -> None:
        title = "Ajouter au fichier général" if self.target.member_id is None else "Ajouter à ta fiche"
        await interaction.response.send_modal(TextModal(
            self, title, "Lignes à ajouter", self.store.add_lines,
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
        await interaction.response.defer()  # voir ANSWER_FIRST
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
        await interaction.response.defer()  # voir ANSWER_FIRST
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
                    f"aux autres ; à part toi, seuls les administrateurs du serveur peuvent la voir, en "
                    f"supprimer des lignes ou l'effacer. Avec /memoire, tu pourras la voir, y ajouter, modifier ou supprimer des lignes, ou désactiver la mémoire.")
        await confirm(interaction, self, question, self.enable)

    async def enable(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()  # voir ANSWER_FIRST
        saved = await self.store.members.set_memory(self.target.member_id, True)
        await self.refresh(interaction, "Mémoire activée." if saved else NOT_SAVED.format(setting="l'activation"))


class OwnerView(discord.ui.LayoutView):
    """Écran réservé à la personne qui a lancé la commande."""

    def __init__(self, owner_id: int):
        super().__init__(timeout=VIEW_TIMEOUT)
        self.owner_id = owner_id
        self.interaction: discord.Interaction | None = None  # celle qui a affiché cet écran (voir show)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id

    async def on_timeout(self) -> None:
        """Boutons grisés : ils ne répondent plus. Le message éphémère reste modifiable 15 minutes après
        l'interaction qui l'a affiché, plus que VIEW_TIMEOUT."""
        if self.interaction is None:
            return
        for item in self.walk_children():
            if isinstance(item, discord.ui.Button):
                item.disabled = True
        self.add_item(discord.ui.TextDisplay(EXPIRED))
        try:
            await self.interaction.edit_original_response(view=self)
        except discord.HTTPException:
            pass  # message fermé par la personne, ou trop ancien

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item) -> None:
        logger.exception("Erreur sur le bouton « %s »", getattr(item, "label", "?"), exc_info=error)
        await reply_error(interaction)


def button(label: str, callback: Callback, style: discord.ButtonStyle = discord.ButtonStyle.secondary,
           emoji: str | None = None, disabled: bool = False) -> discord.ui.Button:
    item = discord.ui.Button(label=label, emoji=emoji, style=style, disabled=disabled)
    item.callback = callback
    return item


def page_buttons(page: int, pages: int, turn: Callable[[discord.Interaction, int], Awaitable[None]],
                 label: bool = True) -> list[discord.ui.Button]:
    """« ◀ », « Page 2/3 » (sans action, si label) et « ▶ »."""
    async def previous(interaction: discord.Interaction) -> None:
        await turn(interaction, page - 1)

    async def following(interaction: discord.Interaction) -> None:
        await turn(interaction, page + 1)
    buttons = [button("◀", previous, disabled=page == 0)]
    if label:
        buttons.append(discord.ui.Button(label=f"Page {page + 1}/{pages}", disabled=True))
    buttons.append(button("▶", following, disabled=page == pages - 1))
    return buttons


async def confirm(interaction: discord.Interaction, card: MemoryCard, question: str, action: Callback) -> None:
    """Remplace le fichier affiché par une question, avec « Confirmer » et « Annuler »."""
    view = OwnerView(card.owner_id)
    view.add_item(discord.ui.Container(discord.ui.TextDisplay(question), accent_colour=discord.Color.orange()))
    view.add_item(discord.ui.ActionRow(button("Confirmer", action, discord.ButtonStyle.danger, "⚠️"),
                                       button("Annuler", card.refresh, emoji="↩️")))
    await card.show(interaction, view)


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
        await interaction.response.defer()  # voir ANSWER_FIRST ; sans réponse, la fenêtre resterait ouverte
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


def numbered(index: int, line: str, chosen: bool) -> str:
    """Ligne de l'écran « Supprimer des lignes » : numérotée, barrée si elle est cochée."""
    text = f"**{index + 1}.** {clip(line)}"
    return f"~~{text}~~" if chosen else text


def clip(line: str) -> str:
    """Ligne coupée si elle ne tient pas seule sur une page (seulement si MEMORY_*_LIMIT le permet)."""
    limit = PAGE_TEXT_MAX - 20  # place pour le numéro et le barré
    return line if len(line) <= limit else line[:limit - 1] + "…"


def paginate(lengths: list[int], text_max: int, per_page: int | None = None) -> list[tuple[int, int]]:
    """Découpe des lignes (leurs longueurs) en pages : (début, fin) de chaque page, des lignes consécutives
    dont le texte, sauts de ligne compris, tient dans text_max, per_page lignes au plus."""
    bounds, start, size = [], 0, 0
    for i, length in enumerate(lengths):
        if i > start and (size + length > text_max or (per_page and i - start >= per_page)):
            bounds.append((start, i))
            start, size = i, 0
        size += length + 1
    bounds.append((start, len(lengths)))
    return bounds


def member_target(store: MemoryStore, member: discord.User | discord.Member, owner_id: int) -> Target:
    entry = store.members.get(member.id)
    name = display_name(entry) if entry else member.display_name
    # Hors du serveur, Discord donne un discord.User au lieu d'un discord.Member.
    is_self = member.id == owner_id
    departed = not is_self and not isinstance(member, discord.Member)
    return Target(store.member_file(member.id), member.id, name, is_self=is_self, departed=departed)


def server_target() -> Target:
    return Target(SERVER_FILE, None)
