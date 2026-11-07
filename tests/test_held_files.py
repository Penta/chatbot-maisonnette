"""Tests du budget de mémoire des fichiers en attente : sans lui, des dizaines de PDF reçus pendant une absence
font tomber le bot, en boucle à chaque reprise. Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import contextlib
import unittest
from datetime import datetime, timezone
from unittest import mock

from bot.attachments import FILES_QUEUE_FULL, HELD_FILES_MAX_BYTES, HeldFiles, readable_size
from bot.claude_client import ClaudeError
from bot.discord_bot import ChatBot, PendingMessage
from bot.media import PDF_MEDIA_TYPE, ImageData

CHANNEL_ID = 10
PDF_SIZE = 15_000_000
BUDGET_IN_PDFS = HELD_FILES_MAX_BYTES // PDF_SIZE


def attachment(size: int, filename: str = "doc.pdf", content_type: str = "application/pdf") -> mock.Mock:
    file = mock.Mock(size=size, content_type=content_type)
    file.filename = filename
    return file


def fake_message(message_id: int, size: int = PDF_SIZE) -> mock.Mock:
    return mock.Mock(id=message_id, channel=mock.Mock(id=CHANNEL_ID), attachments=[attachment(size)], content="lis ça",
                     stickers=[], created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))


def queued(message: mock.Mock, data_size: int = PDF_SIZE) -> PendingMessage:
    line = {"id": message.id, "date": "07/11/2026 03:00", "author": "membre", "content": "lis ça"}
    return PendingMessage(message, line, "", [ImageData(PDF_MEDIA_TYPE, bytes(data_size), name="doc.pdf")])


def fake_bot() -> mock.Mock:
    bot = mock.Mock()
    bot.held_files = HeldFiles()
    bot.settings.reply_pause = 0
    bot._pending, bot._preparing, bot._mentions, bot._waiters, bot._waiting = {}, {}, {}, {}, set()
    bot._answering = set()
    bot._last_main_message = None
    bot.available.return_value = True
    bot.safe_reply = mock.AsyncMock()
    bot.answer_main_channel = mock.AsyncMock()
    bot.answer_mentions = mock.AsyncMock()
    bot.read_main_message = mock.AsyncMock(side_effect=lambda message, content: queued(message))
    bot.history.lock.return_value = asyncio.Lock()
    bot.history.get.return_value = []
    bot.history.add_exchange = mock.AsyncMock()
    bot.history.delete = mock.AsyncMock(return_value=False)
    bot.presence_note.return_value = None
    bot.member_notes = mock.AsyncMock(return_value="")
    bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text="Réponse", web_used=False))
    bot.send_answer = mock.AsyncMock()
    return bot


class HeldFilesTest(unittest.TestCase):
    def test_reserve_up_to_limit(self):
        held = HeldFiles(limit=100)
        self.assertTrue(held.reserve(1, 60))
        self.assertTrue(held.reserve(2, 40))
        self.assertFalse(held.reserve(3, 1))
        self.assertEqual(held.total, 100)

    def test_message_without_file_always_passes(self):
        held = HeldFiles(limit=100)
        held.reserve(1, 100)
        self.assertTrue(held.reserve(2, 0))

    def test_settle_keeps_real_size(self):
        # Une photo réduite pèse bien moins que l'original : la place en trop revient au budget.
        held = HeldFiles(limit=100)
        held.reserve(1, 100)
        held.settle(1, 10)
        self.assertEqual(held.total, 10)
        self.assertTrue(held.reserve(2, 90))

    def test_release_twice_is_harmless(self):
        held = HeldFiles(limit=100)
        held.reserve(1, 100)
        held.release(1)
        held.release(1, 2)
        self.assertEqual(held.total, 0)

    def test_settle_after_release_counts_nothing(self):
        # Message supprimé pendant son téléchargement : sa place ne doit pas revenir pour toujours.
        held = HeldFiles(limit=100)
        held.reserve(1, 50)
        held.release(1)
        held.settle(1, 50)
        self.assertEqual(held.total, 0)

    def test_only_readable_files_count(self):
        # Une vidéo n'est jamais téléchargée : elle ne doit pas faire refuser la mention qui la contient.
        files = [attachment(500_000_000, "film.mp4", "video/mp4"), attachment(1000, "photo.png", "image/png"),
                 attachment(2000), attachment(300, "notes.txt", "text/plain")]
        self.assertEqual(readable_size(files), 3300)


class MainChannelTest(unittest.IsolatedAsyncioTestCase):
    async def test_arrivals_in_parallel_stay_within_budget(self):
        # La reprise au démarrage lit tout en parallèle : la place est réservée avant chaque téléchargement.
        bot = fake_bot()
        downloading = 0
        most = 0

        async def read(message, content):
            nonlocal downloading, most
            downloading += 1
            most = max(most, downloading)
            await asyncio.sleep(0)
            downloading -= 1
            return queued(message)
        bot.read_main_message = read
        await asyncio.gather(*(ChatBot.handle_main_channel(bot, fake_message(i)) for i in range(60)))
        self.assertEqual(most, BUDGET_IN_PDFS)
        self.assertEqual(len(bot._pending[CHANNEL_ID]), BUDGET_IN_PDFS)
        self.assertEqual(bot.safe_reply.await_count, 60 - BUDGET_IN_PDFS)
        self.assertLessEqual(bot.held_files.total, HELD_FILES_MAX_BYTES)

    async def test_refused_without_download(self):
        bot = fake_bot()
        bot.held_files.reserve(999, HELD_FILES_MAX_BYTES)
        message = fake_message(1)
        await ChatBot.handle_main_channel(bot, message)
        bot.read_main_message.assert_not_awaited()
        bot.safe_reply.assert_awaited_once_with(message, FILES_QUEUE_FULL)
        bot.remember_waiting.assert_not_called()
        self.assertEqual(bot._pending, {})

    async def test_text_message_passes_when_full(self):
        bot = fake_bot()
        bot.held_files.reserve(999, HELD_FILES_MAX_BYTES)
        message = fake_message(1)
        message.attachments = []
        await ChatBot.handle_main_channel(bot, message)
        bot.safe_reply.assert_not_awaited()
        self.assertEqual(len(bot._pending[CHANNEL_ID]), 1)

    async def test_real_size_kept_once_read(self):
        bot = fake_bot()
        bot.read_main_message = mock.AsyncMock(side_effect=lambda message, content: queued(message, data_size=1000))
        await ChatBot.handle_main_channel(bot, fake_message(1))
        self.assertEqual(bot.held_files.total, 1000)

    async def test_place_released_when_message_not_queued(self):
        bot = fake_bot()
        bot.read_main_message = mock.AsyncMock(return_value=None)  # pièce jointe refusée, ou rien à lire
        await ChatBot.handle_main_channel(bot, fake_message(1))
        self.assertEqual(bot.held_files.total, 0)

    async def test_place_released_when_reading_fails(self):
        bot = fake_bot()
        bot.read_main_message = mock.AsyncMock(side_effect=RuntimeError("panne"))
        with self.assertLogs("chatbot.discord", "ERROR"):
            await ChatBot.handle_main_channel(bot, fake_message(1))
        self.assertEqual(bot.held_files.total, 0)

    async def test_place_released_when_message_deleted(self):
        bot = fake_bot()
        await ChatBot.handle_main_channel(bot, fake_message(1))
        await ChatBot.forget_deleted(bot, CHANNEL_ID, {1})
        self.assertEqual(bot.held_files.total, 0)


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class AnswerReleasesTest(unittest.IsolatedAsyncioTestCase):
    async def answer(self, bot: mock.Mock) -> None:
        channel = mock.Mock(id=CHANNEL_ID)
        message = fake_message(1)
        bot.held_files.reserve(1, PDF_SIZE)
        bot._pending[CHANNEL_ID] = [queued(message)]
        await ChatBot.answer_main_channel(bot, channel)

    async def test_released_after_answer(self):
        bot = fake_bot()
        await self.answer(bot)
        self.assertEqual(bot.held_files.total, 0)

    async def test_released_after_claude_error(self):
        bot = fake_bot()
        bot.claude.ask.side_effect = ClaudeError("Erreur.")
        await self.answer(bot)
        self.assertEqual(bot.held_files.total, 0)

    async def test_released_when_preparing_the_call_fails(self):
        bot = fake_bot()
        bot.presence_note.side_effect = RuntimeError("panne")
        with self.assertRaises(RuntimeError):
            await self.answer(bot)
        self.assertEqual(bot.held_files.total, 0)


class MentionTest(unittest.IsolatedAsyncioTestCase):
    async def test_refused_when_full(self):
        bot = fake_bot()
        bot.held_files.reserve(999, HELD_FILES_MAX_BYTES)
        message = fake_message(1)
        await ChatBot.handle_mention(bot, message)
        bot.safe_reply.assert_awaited_once_with(message, FILES_QUEUE_FULL)
        bot.answer_mentions.assert_not_awaited()
        self.assertEqual(bot._mentions, {})

    async def test_place_held_until_answered(self):
        bot = fake_bot()
        message = fake_message(1)
        await ChatBot.handle_mention(bot, message)
        self.assertEqual(bot.held_files.total, PDF_SIZE)
        bot.answer_mention = mock.AsyncMock()
        await ChatBot.answer_mentions(bot, message.channel)
        self.assertEqual(bot.held_files.total, 0)

    async def test_released_when_answer_fails(self):
        bot = fake_bot()
        message = fake_message(1)
        await ChatBot.handle_mention(bot, message)
        bot.answer_mention = mock.AsyncMock(side_effect=RuntimeError("panne"))
        with self.assertRaises(RuntimeError):
            await ChatBot.answer_mentions(bot, message.channel)
        self.assertEqual(bot.held_files.total, 0)

    async def test_place_released_when_mention_deleted(self):
        bot = fake_bot()
        await ChatBot.handle_mention(bot, fake_message(1))
        await ChatBot.forget_deleted(bot, CHANNEL_ID, {1})
        self.assertEqual(bot.held_files.total, 0)


if __name__ == "__main__":
    unittest.main()
