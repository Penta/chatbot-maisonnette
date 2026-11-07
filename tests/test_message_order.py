"""Tests de l'ordre des réponses dans le salon dédié : un message dont les pièces jointes sont encore en préparation
n'est pas doublé par les suivants. Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import contextlib
import unittest
from datetime import datetime, timezone
from unittest import mock

from bot.attachments import HeldFiles
from bot.discord_bot import UNEXPECTED_ERROR, ChatBot, PendingMessage

CHANNEL_ID = 10


def text(message_id: int) -> str:
    return f"07/11/2026 18:00, membre a écrit : message {message_id}"


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class MessageOrderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = bot = mock.Mock()
        bot.settings.reply_pause = 0
        bot.held_files = HeldFiles()
        bot._pending, bot._preparing, bot._waiters = {}, {}, {}
        bot._last_main_message = None
        bot.available.return_value = True
        bot.safe_reply = mock.AsyncMock()
        bot.send_answer = mock.AsyncMock()
        bot.answer_main_channel = lambda channel, late=None: ChatBot.answer_main_channel(bot, channel, late)
        bot.history.lock.return_value = asyncio.Lock()
        bot.history.get.return_value = []
        bot.history.add_exchange = mock.AsyncMock()
        bot.presence_note.return_value = None
        bot.member_notes = mock.AsyncMock(return_value="")
        bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text="Réponse", web_used=False))
        self.channel = mock.Mock(id=CHANNEL_ID)
        self.slow: dict[int, asyncio.Event] = {}  # messages dont la préparation attend le signal du test
        self.results: dict[int, object] = {}  # résultat de préparation différent du message prêt
        bot.read_main_message = self.read

    async def read(self, message: mock.Mock, content: str) -> PendingMessage | None:
        if message.id in self.slow:
            await self.slow[message.id].wait()
        result = self.results.get(message.id, "prêt")
        if isinstance(result, Exception):
            raise result
        if result is None:
            return None
        line = {"id": message.id, "date": "07/11/2026 18:00", "author": "membre", "content": f"message {message.id}"}
        return PendingMessage(message, line, "", [])

    def arrive(self, message_id: int) -> asyncio.Task:
        message = mock.Mock(id=message_id, channel=self.channel, attachments=[], content=f"message {message_id}",
                            stickers=[], created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))
        return asyncio.create_task(ChatBot.handle_main_channel(self.bot, message))

    def asked(self) -> list[list[str]]:
        return [[part for part, _ in call.args[1]] for call in self.bot.claude.ask.await_args_list]

    async def test_text_waits_for_older_image(self):
        # Une capture d'écran, puis « tu en penses quoi ? » : Claude doit voir les deux ensemble.
        self.slow[1] = asyncio.Event()
        image = self.arrive(1)
        await asyncio.sleep(0)
        await self.arrive(2)
        self.bot.claude.ask.assert_not_awaited()
        self.slow[1].set()
        await image
        self.assertEqual(self.asked(), [[text(1), text(2)]])

    async def test_older_message_does_not_wait_for_newer_image(self):
        self.bot.claude.ask.side_effect = None
        self.slow[2] = asyncio.Event()
        await self.arrive(1)
        self.assertEqual(self.asked(), [[text(1)]])
        image = self.arrive(2)
        await asyncio.sleep(0)
        await self.arrive(3)
        self.assertEqual(self.asked(), [[text(1)]])
        self.slow[2].set()
        await image
        self.assertEqual(self.asked(), [[text(1)], [text(2), text(3)]])

    async def test_text_waits_for_newer_image_still_preparing(self):
        # Discord a reçu le texte avant l'image, plus longue à envoyer : le texte l'attend aussi.
        self.slow[2] = asyncio.Event()
        first = self.arrive(1)
        second = self.arrive(2)
        await first
        self.bot.claude.ask.assert_not_awaited()
        self.slow[2].set()
        await second
        self.assertEqual(self.asked(), [[text(1), text(2)]])

    async def test_all_wait_until_every_preparation_ends(self):
        self.slow[1] = asyncio.Event()
        self.slow[2] = asyncio.Event()
        first, second = self.arrive(1), self.arrive(2)
        await asyncio.sleep(0)
        await self.arrive(3)
        self.slow[1].set()
        await first
        self.bot.claude.ask.assert_not_awaited()
        self.slow[2].set()
        await second
        self.assertEqual(self.asked(), [[text(1), text(2), text(3)]])

    async def test_messages_during_pause_get_one_answer(self):
        self.bot.settings.reply_pause = 0.2
        first = self.arrive(1)
        await asyncio.sleep(0.05)
        self.bot.claude.ask.assert_not_awaited()  # le bot n'a pas encore répondu
        second = self.arrive(2)
        await asyncio.gather(first, second)
        self.assertEqual(self.asked(), [[text(1), text(2)]])

    async def test_message_after_pause_gets_its_own_answer(self):
        self.bot.settings.reply_pause = 0.05
        await self.arrive(1)
        await self.arrive(2)
        self.assertEqual(self.asked(), [[text(1)], [text(2)]])

    async def test_preparation_time_counts_in_pause(self):
        # Une image déjà longue à préparer ne fait pas attendre 2 secondes de plus.
        self.bot.settings.reply_pause = 0.05
        self.slow[1] = asyncio.Event()
        with mock.patch("bot.discord_bot.asyncio.sleep", wraps=asyncio.sleep) as sleep:
            image = self.arrive(1)
            await asyncio.wait([image], timeout=0.1)
            self.slow[1].set()
            await image
        self.assertEqual(sleep.await_args.args[0], 0)
        self.assertEqual(self.asked(), [[text(1)]])

    async def test_absent_during_pause_defers_answer(self):
        self.bot.settings.reply_pause = 0.05
        task = self.arrive(1)
        await asyncio.sleep(0)
        self.bot.available.return_value = False
        await task
        self.bot.answer_later.assert_called_once()
        self.bot.claude.ask.assert_not_awaited()

    async def test_refused_attachment_does_not_block_next_message(self):
        self.slow[1] = asyncio.Event()
        self.results[1] = None  # pièce jointe refusée : l'auteur a été prévenu
        refused = self.arrive(1)
        await asyncio.sleep(0)
        await self.arrive(2)
        self.slow[1].set()
        await refused
        self.assertEqual(self.asked(), [[text(2)]])

    async def test_failed_preparation_does_not_block_next_message(self):
        self.slow[1] = asyncio.Event()
        self.results[1] = RuntimeError("panne")
        failed = self.arrive(1)
        await asyncio.sleep(0)
        await self.arrive(2)
        self.slow[1].set()
        with self.assertLogs("chatbot.discord", "ERROR"):
            await failed
        self.assertEqual(self.asked(), [[text(2)]])
        self.assertEqual(self.bot.safe_reply.await_args.args[1], UNEXPECTED_ERROR)
        self.assertEqual(self.bot.held_files.total, 0)

    async def test_absent_message_waits_for_return(self):
        self.bot.available.return_value = False
        await self.arrive(1)
        self.bot.answer_later.assert_called_once()
        self.bot.claude.ask.assert_not_awaited()

    async def test_absent_after_refused_attachment_defers_the_others(self):
        # Alice écrit pendant que l'image de Paul se prépare, Rika s'absente, puis l'image est refusée.
        self.slow[1] = asyncio.Event()
        self.results[1] = None
        paul = self.arrive(1)
        await asyncio.sleep(0)
        await self.arrive(2)
        self.bot.available.return_value = False
        self.slow[1].set()
        await paul
        self.assertEqual([call.args[0].id for call in self.bot.answer_later.call_args_list], [2])

    async def test_absent_after_message_answered_not_deferred(self):
        # Un autre message a emporté celui-ci pendant la pause, avant que Rika s'absente : plus rien n'attend.
        self.bot.settings.reply_pause = 0.05
        task = self.arrive(1)
        await asyncio.sleep(0)
        self.bot._pending[CHANNEL_ID].clear()
        self.bot.available.return_value = False
        await task
        self.bot.answer_later.assert_not_called()

    async def test_absent_refused_attachment_not_deferred(self):
        self.bot.available.return_value = False
        self.results[1] = None
        await self.arrive(1)
        self.bot.answer_later.assert_not_called()


if __name__ == "__main__":
    unittest.main()
