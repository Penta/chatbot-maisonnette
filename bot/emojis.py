"""Emojis du serveur : Claude les écrit par leur nom (:hap:), Discord les reçoit sous forme de code."""

from __future__ import annotations

import re

import discord

# Code Discord d'un emoji personnalisé : <:nom:123…>, ou <a:nom:123…> s'il est animé.
DISCORD_EMOJI = re.compile(r"<a?:(\w+):\d+>")
# Nom d'emoji entre deux-points ; le code est repris tel quel, comme les blocs de code.
_SHORT_EMOJI = re.compile(r"```.*?```|`[^`\n]*`|<a?:\w+:\d+>|:([A-Za-z0-9_]{2,32}):", re.S)
SHORT_EMOJI = re.compile(r":[A-Za-z0-9_]{2,32}:")


def usable_emojis(guild: discord.Guild | None) -> list[discord.Emoji]:
    """Emojis que le bot peut envoyer, triés par nom (la liste du prompt reste ainsi en cache).

    Animés compris : un bot n'a pas besoin de Nitro. Ceux réservés à des rôles que le bot
    n'a pas sont exclus."""
    if guild is None:
        return []
    roles = set(guild.me.roles) if guild.me else set()
    emojis = [e for e in guild.emojis if e.available and (not e.roles or roles.intersection(e.roles))]
    return sorted(emojis, key=lambda e: (e.name.lower(), e.id))


def emoji_names(guild: discord.Guild | None) -> list[str]:
    return [e.name for e in usable_emojis(guild)]


def short_emojis(text: str) -> str:
    """Remplace les codes des emojis (<:hap:123…>) par leur nom (:hap:), plus court à lire pour Claude."""
    return DISCORD_EMOJI.sub(r":\1:", text)


def discord_emojis(text: str, guild: discord.Guild | None) -> str:
    """Remplace les noms d'emojis écrits par Claude (:hap:) par leur code Discord.

    Un nom inconnu reste tel quel ; la casse est tolérée (:Hap: pour :hap:)."""
    emojis = usable_emojis(guild)
    if not emojis:
        return text
    exact = {e.name: e for e in emojis}
    loose = {e.name.lower(): e for e in reversed(emojis)}

    def replace(match: re.Match) -> str:
        name = match.group(1)
        emoji = name and (exact.get(name) or loose.get(name.lower()))
        return str(emoji) if emoji else match.group(0)

    return _SHORT_EMOJI.sub(replace, text)
