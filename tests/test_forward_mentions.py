"""Tests des mentions dans un message transféré : Rika les lit comme dans un message normal.
Lancer depuis la racine du projet : python -m unittest"""

import unittest
from unittest import mock

from bot.discord_bot import ChatBot


class ForwardMentionsTest(unittest.TestCase):
    def setUp(self):
        self.bot = mock.Mock()
        paulo = mock.Mock(display_name="Paulo")
        modos = mock.Mock()
        modos.name = "Modérateurs"
        general = mock.Mock()
        general.name = "général"
        self.guild = mock.Mock()
        self.guild.get_member.side_effect = {1: paulo}.get
        self.guild.get_role.side_effect = {5: modos}.get
        self.bot.get_user.side_effect = {1: mock.Mock(display_name="paul_42"), 2: mock.Mock(display_name="Léa")}.get
        self.bot.get_channel.side_effect = {7: general}.get

    def clean(self, text: str, guild=None) -> str:
        return ChatBot.clean_mentions(self.bot, text, guild)

    def test_member_named_by_server_nickname(self):
        # Paul s'appelle « Paulo » sur le serveur et « paul_42 » ailleurs : Rika voit le nom qu'elle connaît.
        self.assertEqual(self.clean("salut <@1> et <@!1>", self.guild), "salut @Paulo et @Paulo")

    def test_user_outside_server_keeps_global_name(self):
        self.assertEqual(self.clean("salut <@2>", self.guild), "salut @Léa")
        self.assertEqual(self.clean("salut <@1>"), "salut @paul_42")  # message privé : pas de serveur

    def test_unknown_user(self):
        self.assertEqual(self.clean("salut <@3>", self.guild), "salut @quelqu'un")

    def test_roles_and_channels(self):
        self.assertEqual(self.clean("<@&5> voir <#7>", self.guild), "@Modérateurs voir #général")

    def test_unknown_role_and_channel(self):
        # Transfert venu d'un autre serveur : son rôle et son salon restent inconnus, sans code illisible.
        self.assertEqual(self.clean("<@&6> voir <#8>", self.guild), "@rôle voir #salon")
        self.assertEqual(self.clean("<@&5>"), "@rôle")


if __name__ == "__main__":
    unittest.main()
