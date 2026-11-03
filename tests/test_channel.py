"""Tests de la vérification du salon dédié (CHANNEL_ID). Lancer depuis la racine du projet : python -m unittest"""

import unittest
from unittest import mock

import discord

from bot.discord_bot import ChatBot, channel_problem

CHANNEL_ID, GUILD_ID = 10, 1000


def fake_client(channel=None, guild=None) -> mock.Mock:
    client = mock.Mock()
    client.get_channel.return_value = channel
    client.get_guild.return_value = guild
    return client


def fake_channel(cls=discord.TextChannel, **permissions) -> mock.Mock:
    channel = mock.Mock(spec=cls)
    channel.guild = mock.Mock(id=GUILD_ID)
    channel.__str__ = lambda self: "salon"
    allowed = {"view_channel": True, "send_messages": True, "send_messages_in_threads": True,
               "read_message_history": True} | permissions
    channel.permissions_for.return_value = mock.Mock(**allowed)
    return channel


class ChannelProblemTest(unittest.TestCase):
    def test_usable_channel(self):
        self.assertIsNone(channel_problem(fake_client(fake_channel()), CHANNEL_ID))

    def test_unknown_channel(self):
        self.assertIn("introuvable", channel_problem(fake_client(), CHANNEL_ID))

    def test_server_id_instead_of_channel(self):
        self.assertIn("identifiant du serveur", channel_problem(fake_client(guild=mock.Mock()), CHANNEL_ID))

    def test_category_and_forum_refused(self):
        self.assertIn("une catégorie", channel_problem(fake_client(fake_channel(discord.CategoryChannel)), CHANNEL_ID))
        self.assertIn("un forum", channel_problem(fake_client(fake_channel(discord.ForumChannel)), CHANNEL_ID))

    def test_missing_permissions(self):
        problem = channel_problem(fake_client(fake_channel(send_messages=False)), CHANNEL_ID)
        self.assertIn("la permission « Envoyer des messages »", problem)
        problem = channel_problem(fake_client(fake_channel(view_channel=False, read_message_history=False)), CHANNEL_ID)
        self.assertIn("les permissions « Voir les salons », « Voir les anciens messages »", problem)

    def test_thread_uses_thread_permission(self):
        thread = fake_channel(discord.Thread, send_messages=False)
        self.assertIsNone(channel_problem(fake_client(thread), CHANNEL_ID))
        thread = fake_channel(discord.Thread, send_messages_in_threads=False)
        self.assertIn("Envoyer des messages dans les fils", channel_problem(fake_client(thread), CHANNEL_ID))


class AnnounceTest(unittest.IsolatedAsyncioTestCase):
    def fake_bot(self, *, unavailable: bool) -> mock.Mock:
        bot = mock.Mock(_announced=False, config_error=None)
        bot.settings.channel_id = CHANNEL_ID
        bot.get_channel.return_value = None
        bot.get_guild.return_value = None
        bot.guilds = [mock.Mock(unavailable=unavailable)]
        bot.close = mock.AsyncMock()
        return bot

    async def test_wrong_channel_stops_bot(self):
        bot = self.fake_bot(unavailable=False)
        await ChatBot.announce(bot)
        self.assertIn("introuvable", bot.config_error)
        bot.close.assert_awaited_once()

    async def test_unavailable_server_waits(self):
        # Panne Discord : le salon n'est pas encore connu, mais le bot reste allumé.
        bot = self.fake_bot(unavailable=True)
        await ChatBot.announce(bot)
        self.assertIsNone(bot.config_error)
        self.assertFalse(bot._announced)
        bot.close.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
