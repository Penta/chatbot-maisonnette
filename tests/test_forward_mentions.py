"""Tests des mentions dans un message transféré : Rika les lit comme dans un message normal.
Lancer depuis la racine du projet : python -m unittest"""

import unittest
from unittest import mock

import discord

from bot.discord_bot import ChatBot


def not_found(*args):
    raise discord.NotFound(mock.Mock(status=404, reason="Not Found"), "inconnu")


class ForwardMentionsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Comme en production : le bot ne garde aucun membre ni utilisateur en mémoire (pas d'option « membres »).
        self.bot = mock.Mock()
        self.bot.get_user.return_value = None
        paulo = mock.Mock(display_name="Paulo")
        modos = mock.Mock()
        modos.name = "Modérateurs"
        general = mock.Mock()
        general.name = "général"
        self.guild = mock.Mock()
        self.guild.get_member.return_value = None
        self.guild.fetch_member = mock.AsyncMock(side_effect=lambda user_id: {1: paulo}.get(user_id) or not_found())
        self.guild.get_role.side_effect = {5: modos}.get
        users = {1: mock.Mock(display_name="paul_42"), 2: mock.Mock(display_name="Léa")}
        self.bot.fetch_user = mock.AsyncMock(side_effect=lambda user_id: users.get(user_id) or not_found())
        self.bot.get_channel.side_effect = {7: general}.get
        self.bot.mentioned_name = lambda user_id, guild: ChatBot.mentioned_name(self.bot, user_id, guild)

    async def clean(self, text: str, guild=None) -> str:
        return await ChatBot.clean_mentions(self.bot, text, guild)

    async def test_member_named_by_server_nickname(self):
        # Paul s'appelle « Paulo » sur le serveur et « paul_42 » ailleurs : Rika voit le nom qu'elle connaît.
        self.assertEqual(await self.clean("salut <@1> et <@!1>", self.guild), "salut @Paulo et @Paulo")
        self.guild.fetch_member.assert_awaited_once_with(1)  # une seule demande à Discord par personne

    async def test_member_in_memory_not_asked(self):
        self.guild.get_member.return_value = mock.Mock(display_name="Paulo")
        self.assertEqual(await self.clean("salut <@1>", self.guild), "salut @Paulo")
        self.guild.fetch_member.assert_not_awaited()

    async def test_nickname_even_if_user_in_memory(self):
        # Le bot se souvient de « paul_42 », mais Paul est membre du serveur : Rika voit son surnom.
        self.bot.get_user.return_value = mock.Mock(display_name="paul_42")
        self.assertEqual(await self.clean("salut <@1>", self.guild), "salut @Paulo")

    async def test_user_outside_server_keeps_global_name(self):
        self.assertEqual(await self.clean("salut <@2>", self.guild), "salut @Léa")
        self.assertEqual(await self.clean("salut <@1>"), "salut @paul_42")  # message privé : pas de serveur

    async def test_unknown_user(self):
        self.assertEqual(await self.clean("salut <@3>", self.guild), "salut @quelqu'un")

    async def test_discord_failure_keeps_someone(self):
        self.guild.fetch_member.side_effect = discord.HTTPException(mock.Mock(status=500, reason="Erreur"), "panne")
        self.assertEqual(await self.clean("salut <@1>", self.guild), "salut @quelqu'un")

    async def test_roles_and_channels(self):
        self.assertEqual(await self.clean("<@&5> voir <#7>", self.guild), "@Modérateurs voir #général")

    async def test_unknown_role_and_channel(self):
        # Transfert venu d'un autre serveur : son rôle et son salon restent inconnus, sans code illisible.
        self.assertEqual(await self.clean("<@&6> voir <#8>", self.guild), "@rôle voir #salon")
        self.assertEqual(await self.clean("<@&5>"), "@rôle")


if __name__ == "__main__":
    unittest.main()
