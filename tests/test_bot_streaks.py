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
        self.bot.handle_main_channel = mock.AsyncMock()
        self.bot.register_members = mock.AsyncMock()

    async def receive(self, from_bot: bool, when: datetime = START) -> None:
        message = mock.Mock(guild=mock.Mock(id=HOME), webhook_id=None, type=discord.MessageType.reply,
                            created_at=when)
        message.channel.id = CHANNEL_ID
        message.author.bot = from_bot
        message.mentions = [self.bot.user]  # « Répondre » avec ping
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


if __name__ == "__main__":
    unittest.main()
