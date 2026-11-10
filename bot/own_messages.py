"""Messages du bot : outil « mes_messages » de Claude, pour modifier ou supprimer ses propres messages récents."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import discord

from .emojis import discord_emojis, short_emojis
from .errors import DISCORD_FAILURES
from .memory import ToolResult
from .text_utils import DISCORD_LIMIT

logger = logging.getLogger("chatbot.discord")

# Derniers messages du salon, tous auteurs confondus, parmi lesquels le bot cherche le sien.
SEARCHED_MESSAGES = 50
# Modifications et suppressions au plus par réponse.
MAX_MESSAGE_ACTIONS = 3

# Le message se désigne par un passage recopié, comme « ancien » dans l'outil « memoire » : Claude ne voit
# pas les identifiants des messages, et sa réponse peut avoir été découpée en plusieurs messages Discord.
OWN_MESSAGES_TOOL = {
    "name": "mes_messages",
    "description": (
        "Modifie ou supprime l'un de tes propres messages récents dans ce salon (jamais ceux des autres), "
        "par exemple pour cacher un spoiler sous ||balise spoiler|| ou corriger une erreur. « passage » : "
        "un extrait de ce message recopié exactement, qui ne figure dans aucun autre de tes messages récents. "
        "« modifier » remplace ce passage par « texte » (vide pour le retirer) ; « supprimer » efface tout le "
        "message, et « texte » reste vide."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["modifier", "supprimer"]},
            "passage": {"type": "string"},
            "texte": {"type": "string"},
        },
        "required": ["action", "passage", "texte"],
        "additionalProperties": False,
    },
    "strict": True,
}


class OwnMessages:
    """Exécute l'outil « mes_messages » dans un salon : seuls les messages du bot lui-même y sont touchés.

    on_edit(identifiant, texte avant, texte après) : appelé après une modification, pour que la mémoire du salon
    garde le nouveau texte du message (None hors du salon dédié, qui n'a pas de mémoire)."""

    def __init__(self, channel: discord.abc.Messageable, me: discord.abc.User,
                 on_edit: Callable[[int, str, str], Awaitable[None]] | None = None):
        self.channel = channel
        self.me = me
        self.on_edit = on_edit

    async def run(self, command: dict) -> ToolResult:
        action, passage, text = command.get("action"), command.get("passage") or "", command.get("texte") or ""
        if action not in ("modifier", "supprimer"):
            return ToolResult(f"Action inconnue : {action}.", is_error=True)
        if not passage.strip():
            return ToolResult("« passage » est vide : recopie un extrait du message.", is_error=True)
        try:
            found = [message async for message in self.channel.history(limit=SEARCHED_MESSAGES)
                     if message.author.id == self.me.id and passage in short_emojis(message.content)]
        except discord.Forbidden:
            return ToolResult("Tu n'as pas la permission de relire ce salon.", is_error=True)
        except DISCORD_FAILURES as exc:
            logger.warning("Messages de #%s illisibles, outil « mes_messages » sans effet : %s", self.channel, exc)
            return ToolResult("Discord ne répond pas, impossible pour l'instant.", is_error=True)
        if not found:
            return ToolResult("Aucun de tes messages récents dans ce salon ne contient ce passage, recopié "
                              "exactement.", is_error=True)
        if len(found) > 1:
            return ToolResult("Ce passage figure dans plusieurs de tes messages : recopie un passage plus long.",
                              is_error=True)
        message = found[0]
        if action == "supprimer":
            return await self._delete(message)
        current = short_emojis(message.content)
        if current.count(passage) > 1:
            return ToolResult("Ce passage figure plusieurs fois dans le message : recopie un passage plus long.",
                              is_error=True)
        revised = current.replace(passage, text)
        if not revised.strip():
            return ToolResult("Le message serait vide : supprime-le plutôt.", is_error=True)
        content = discord_emojis(revised, getattr(self.channel, "guild", None))
        if len(content) > DISCORD_LIMIT:
            return ToolResult(f"Le message dépasserait {DISCORD_LIMIT} caractères : raccourcis « texte ».",
                              is_error=True)
        try:
            await message.edit(content=content)
        except DISCORD_FAILURES as exc:
            logger.warning("Modification du message %s impossible : %s", message.id, exc)
            return ToolResult("Discord a refusé la modification.", is_error=True)
        logger.info("Message %s du bot modifié dans #%s.", message.id, self.channel)
        if self.on_edit:
            await self.on_edit(message.id, current, revised)
        return ToolResult("Message modifié.")

    async def _delete(self, message: discord.Message) -> ToolResult:
        try:
            await message.delete()
        except discord.NotFound:
            return ToolResult("Ce message est déjà supprimé.")
        except DISCORD_FAILURES as exc:
            logger.warning("Suppression du message %s impossible : %s", message.id, exc)
            return ToolResult("Discord a refusé la suppression.", is_error=True)
        logger.info("Message %s du bot supprimé dans #%s.", message.id, self.channel)
        return ToolResult("Message supprimé.")
