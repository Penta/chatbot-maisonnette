"""Tests de l'ordre des réponses dans le salon dédié : un message dont les pièces jointes sont encore en préparation
n'est pas doublé par les suivants. Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import contextlib
import unittest
from datetime import datetime, timezone
from unittest import mock

from bot.attachments import HeldFiles
from bot.discord_bot import DELETED_KEPT, UNEXPECTED_ERROR, ChatBot, PendingMessage

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
        bot._pending, bot._preparing, bot._waiters, bot._mentions = {}, {}, {}, {}
        bot._deleted, bot._waiting = set(), set()
        bot.history.delete = mock.AsyncMock(return_value=False)
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

    async def delete(self, *message_ids: int) -> None:
        await ChatBot.forget_deleted(self.bot, CHANNEL_ID, set(message_ids))

    def deleted_while_asking(self, *message_ids: int) -> None:
        """Ces messages sont supprimés pendant que Claude écrit sa réponse."""
        async def ask(*args, **kwargs):
            await self.delete(*message_ids)
            return mock.Mock(text="Réponse", web_used=False)
        self.bot.claude.ask.side_effect = ask

    def asked(self) -> list[list[str]]:
        return [[part for part, _ in call.args[1]] for call in self.bot.claude.ask.await_args_list]

    async def test_sticker_or_emoji_alone_answered_by_claude(self):
        # Paul envoie un sticker seul, puis un emoji du serveur seul : Claude reçoit chaque message et y répond
        # avec des mots, même si le serveur a des stickers et des emojis qu'il pourrait renvoyer.
        bot = self.bot
        bot.read_main_message = ChatBot.read_main_message.__get__(bot)
        bot.read_forward = mock.AsyncMock(return_value=(None, []))
        bot.read_attachments = mock.AsyncMock(return_value=([], []))
        bot.read_reply = mock.AsyncMock(return_value=(None, []))
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.message_body = ChatBot.message_body
        bot.timestamp.return_value = "07/11/2026 18:00"
        sticker = mock.Mock(available=True)
        sticker.name = "chat qui pleure"
        guild = mock.Mock(stickers=[sticker])
        for content, stickers, seen in (("", [sticker], "[sticker « chat qui pleure »]"), ("<:rire:1>", [], ":rire:")):
            with self.subTest(seen=seen):
                bot.claude.ask.reset_mock()
                self.channel.send = mock.AsyncMock()
                message = mock.Mock(id=1, channel=self.channel, guild=guild, attachments=[], content=content,
                                    clean_content=content, stickers=stickers,
                                    created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))
                message.author.display_name = message.author.name = "membre"
                with mock.patch("bot.discord_bot.usable_emojis", lambda guild: ["<:rire:1>"]):
                    await ChatBot.handle_main_channel(bot, message)
                self.assertEqual(self.asked(), [[f"07/11/2026 18:00, membre a écrit : {seen}"]])
                self.channel.send.assert_not_awaited()
                bot.send_answer.assert_awaited_with(message, "Réponse", reply=False)

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

    async def test_deleted_while_preparing_not_answered(self):
        # Alice poste une image puis supprime son message pendant le téléchargement : pas de réponse, rien en mémoire.
        self.slow[1] = asyncio.Event()
        task = self.arrive(1)
        await asyncio.sleep(0)
        await self.delete(1)
        self.slow[1].set()
        await task
        self.bot.claude.ask.assert_not_awaited()
        self.bot.history.add_exchange.assert_not_awaited()

    async def test_deleted_while_preparing_others_still_answered(self):
        self.slow[1] = asyncio.Event()
        task = self.arrive(1)
        await asyncio.sleep(0)
        await self.arrive(2)
        await self.delete(1)
        self.slow[1].set()
        await task
        self.assertEqual(self.asked(), [[text(2)]])

    async def test_deleted_before_preparation_not_answered(self):
        # Supprimé pendant que Rika met à jour le registre des membres, avant de commencer à le lire.
        await self.delete(1)
        await self.arrive(1)
        self.bot.claude.ask.assert_not_awaited()
        self.bot.history.add_exchange.assert_not_awaited()

    async def test_deleted_list_keeps_most_recent(self):
        await self.delete(*range(1, DELETED_KEPT + 2))
        self.assertEqual(self.bot._deleted, set(range(2, DELETED_KEPT + 2)))

    async def test_deleted_during_answer_not_sent(self):
        self.deleted_while_asking(1)
        await self.arrive(1)
        self.bot.send_answer.assert_not_awaited()
        self.bot.history.add_exchange.assert_not_awaited()

    async def test_one_deleted_during_answer_others_answered(self):
        self.slow[1] = asyncio.Event()
        task = self.arrive(1)
        await asyncio.sleep(0)
        await self.arrive(2)
        self.deleted_while_asking(2)
        self.slow[1].set()
        await task
        self.bot.history.add_exchange.assert_awaited_once()
        self.bot.send_answer.assert_awaited_once()
        self.assertEqual(self.bot.send_answer.await_args.args[0].id, 1)  # la réponse suit le message restant


if __name__ == "__main__":
    unittest.main()
