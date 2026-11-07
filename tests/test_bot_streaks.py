"""Tests des conversations avec d'autres bots : pas de boucle sans fin. Lancer depuis la racine du projet : python -m unittest"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import discord

from bot.discord_bot import BOT_STREAK_RESET, MAX_BOT_REPLIES, BotStreaks, ChatBot

HOME = 1000
CHANNEL_ID = 10
START = datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc)


class BotStreaksTest(unittest.TestCase):
    def test_counts_per_channel(self):
        streaks = BotStreaks()
        self.assertEqual(streaks.count(1, START), 1)
        self.assertEqual(streaks.count(1, START), 2)
        self.assertEqual(streaks.count(2, START), 1)

    def test_human_resets(self):
        streaks = BotStreaks()
        streaks.count(1, START)
        streaks.human_spoke(1)
        self.assertEqual(streaks.count(1, START), 1)

    def test_quiet_channel_resets(self):
        streaks = BotStreaks()
        streaks.count(1, START)
        self.assertEqual(streaks.count(1, START + BOT_STREAK_RESET), 2)
        self.assertEqual(streaks.count(1, START + 2 * BOT_STREAK_RESET + timedelta(seconds=1)), 1)


class OnMessageTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = mock.Mock()
        self.bot.settings.channel_id = CHANNEL_ID
        self.bot.home_guild_id.return_value = HOME
        self.bot._bot_streaks = BotStreaks()
        self.bot._last_main_message = None
        self.bot.handle_main_channel = mock.AsyncMock()
        self.bot.register_members = mock.AsyncMock()

    async def receive(self, from_bot: bool, when: datetime = START) -> None:
        message = mock.Mock(guild=mock.Mock(id=HOME), webhook_id=None, type=discord.MessageType.reply,
                            created_at=when)
        message.channel.id = CHANNEL_ID
        message.author.bot = from_bot
        message.mentions = [self.bot.user]  # « Répondre » avec ping
        message.raw_mentions = []  # pas de <@id> dans le texte
        await ChatBot.on_message(self.bot, message)

    async def test_bot_loop_stopped(self):
        # Un bot qui répond à chacun de ses messages : le bot s'arrête après MAX_BOT_REPLIES réponses.
        with self.assertLogs("chatbot.discord", "WARNING") as logs:
            for _ in range(MAX_BOT_REPLIES + 3):
                await self.receive(from_bot=True)
        self.assertEqual(self.bot.handle_main_channel.await_count, MAX_BOT_REPLIES)
        self.assertEqual(len(logs.records), 1)  # un seul avertissement par série

    async def test_member_message_restarts(self):
        with self.assertLogs("chatbot.discord", "WARNING"):
            for _ in range(MAX_BOT_REPLIES + 1):
                await self.receive(from_bot=True)
        await self.receive(from_bot=False)
        await self.receive(from_bot=True)
        self.assertEqual(self.bot.handle_main_channel.await_count, MAX_BOT_REPLIES + 2)

    async def test_quiet_channel_restarts(self):
        with self.assertLogs("chatbot.discord", "WARNING"):
            for _ in range(MAX_BOT_REPLIES + 1):
                await self.receive(from_bot=True)
        await self.receive(from_bot=True, when=START + BOT_STREAK_RESET + timedelta(seconds=1))
        self.assertEqual(self.bot.handle_main_channel.await_count, MAX_BOT_REPLIES + 1)


class MentionLogTest(unittest.IsolatedAsyncioTestCase):
    """Journal de diagnostic : un bot qui écrit <@id> du bot, avec ou sans mention transmise par Discord."""

    def setUp(self):
        self.bot = mock.Mock()
        self.bot.settings.channel_id = CHANNEL_ID
        self.bot.home_guild_id.return_value = HOME
        self.bot._bot_streaks = BotStreaks()
        self.bot._last_main_message = None
        self.bot.handle_main_channel = mock.AsyncMock()
        self.bot.register_members = mock.AsyncMock()
        self.bot.replies_to_me = mock.AsyncMock(return_value=False)

    async def receive(self, listed: bool) -> None:
        message = mock.Mock(guild=mock.Mock(id=HOME), webhook_id=None, type=discord.MessageType.default,
                            created_at=START)
        message.channel.id = CHANNEL_ID
        message.author.bot = True
        message.mentions = [self.bot.user] if listed else []
        message.raw_mentions = [self.bot.user.id]
        await ChatBot.on_message(self.bot, message)

    async def test_mention_missing_from_discord_list(self):
        with self.assertLogs("chatbot.discord", "INFO") as logs:
            await self.receive(listed=False)
        self.assertIn("mention absente de la liste de Discord, message ignoré", logs.output[0])
        self.bot.handle_main_channel.assert_not_awaited()

    async def test_mention_listed(self):
        with self.assertLogs("chatbot.discord", "INFO") as logs:
            await self.receive(listed=True)
        self.assertIn("mention transmise par Discord", logs.output[0])
        self.bot.handle_main_channel.assert_awaited_once()

    async def test_no_log_without_mention_in_text(self):
        # « Répondre » à un message du bot, sans <@id> dans le texte : rien de plus dans le journal.
        message = mock.Mock(guild=mock.Mock(id=HOME), webhook_id=None, type=discord.MessageType.reply,
                            created_at=START, mentions=[], raw_mentions=[])
        message.channel.id = CHANNEL_ID
        message.author.bot = True
        self.bot.replies_to_me.return_value = True
        with self.assertNoLogs("chatbot.discord", "INFO"):
            await ChatBot.on_message(self.bot, message)
        self.bot.handle_main_channel.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
