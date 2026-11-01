"""Tests du serveur du bot : rien n'est accepté ailleurs. Lancer depuis la racine du projet : python -m unittest"""

import unittest
from unittest import mock

import discord

from bot.discord_bot import ChatBot, HomeTree

HOME, OTHER = 1000, 2000
CHANNEL_ID = 10


def fake_bot(channel_found: bool = True) -> mock.Mock:
    bot = mock.Mock()
    bot.settings.channel_id = CHANNEL_ID
    bot.get_channel.return_value = mock.Mock(guild=mock.Mock(id=HOME)) if channel_found else None
    bot.home_guild_id = lambda: ChatBot.home_guild_id(bot)
    bot.handle_main_channel = mock.AsyncMock()
    bot.handle_mention = mock.AsyncMock()
    bot.register_members = mock.AsyncMock()
    return bot


def fake_interaction(guild_id: int | None) -> mock.Mock:
    interaction = mock.Mock(guild_id=guild_id, type=discord.InteractionType.application_command,
                            data={"name": "memoire-serveur"})
    interaction.response.send_message = mock.AsyncMock()
    return interaction


class CommandsTest(unittest.IsolatedAsyncioTestCase):
    async def check(self, interaction: mock.Mock, channel_found: bool = True) -> bool:
        tree = object.__new__(HomeTree)  # sans client Discord réel
        tree.client = fake_bot(channel_found)
        return await tree.interaction_check(interaction)

    async def test_home_guild_accepted(self):
        interaction = fake_interaction(HOME)
        self.assertTrue(await self.check(interaction))
        interaction.response.send_message.assert_not_awaited()

    async def test_other_guild_refused(self):
        # Administrateur d'un autre serveur où le bot a été invité : ni lecture ni écriture de la mémoire.
        interaction = fake_interaction(OTHER)
        self.assertFalse(await self.check(interaction))
        interaction.response.send_message.assert_awaited_once()

    async def test_direct_message_refused(self):
        self.assertFalse(await self.check(fake_interaction(None)))

    async def test_refused_while_channel_unknown(self):
        self.assertFalse(await self.check(fake_interaction(HOME), channel_found=False))


class MessagesTest(unittest.IsolatedAsyncioTestCase):
    async def receive(self, guild: mock.Mock | None, channel_id: int = CHANNEL_ID) -> mock.Mock:
        bot = fake_bot()
        message = mock.Mock(guild=guild, webhook_id=None, type=discord.MessageType.default)
        message.channel.id = channel_id
        message.author.bot = False
        message.mentions = [bot.user]
        await ChatBot.on_message(bot, message)
        return bot

    async def test_home_guild_answered(self):
        bot = await self.receive(mock.Mock(id=HOME))
        bot.handle_main_channel.assert_awaited_once()

    async def test_other_guild_ignored(self):
        # Même un salon qui porte le même identifiant ou une mention : rien n'est lu ni noté.
        bot = await self.receive(mock.Mock(id=OTHER))
        bot.handle_main_channel.assert_not_awaited()
        bot.handle_mention.assert_not_awaited()
        bot.register_members.assert_not_awaited()

    async def test_direct_message_ignored(self):
        bot = await self.receive(None, channel_id=99)
        bot.handle_mention.assert_not_awaited()
        bot.register_members.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
