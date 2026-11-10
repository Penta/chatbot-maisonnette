"""Tests du fichier d'attente : un message y reste jusqu'à sa réponse, pour survivre à un redémarrage, sans pouvoir
faire tomber le bot en boucle. Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import contextlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from bot.attachments import HeldFiles
from bot.claude_client import ClaudeError
from bot.discord_bot import ChatBot, PendingMessage
from bot.files import write_atomic

CHANNEL_ID = 10


class WaitingTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.file = Path(folder.name) / "waiting_messages.json"
        self.bot = bot = mock.Mock()
        bot.settings.reply_pause = 0
        bot.settings.context_message_limit = 0
        bot.held_files = HeldFiles()
        bot._waiting_file = self.file
        bot._waiting, bot._replayed, bot._to_replay = set(), set(), set()
        bot._stopping = False
        bot._pending, bot._preparing, bot._mentions, bot._waiters = {}, {}, {}, {}
        bot._deleted = set()
        bot._last_main_message = None
        bot.available.return_value = True
        for name in ("_save_waiting", "remember_waiting", "remember_queued", "forget_waiting", "answer_main_channel",
                     "answer_mentions"):
            setattr(bot, name, getattr(ChatBot, name).__get__(bot))
        bot.read_main_message = mock.AsyncMock(side_effect=self.read)
        bot.mention_part = mock.AsyncMock(return_value=("salut", []))
        bot.mention_context = mock.AsyncMock(return_value=[])
        bot.answer_mention = ChatBot.answer_mention.__get__(bot)
        bot.safe_reply = mock.AsyncMock()
        bot.send_answer = mock.AsyncMock()
        bot.history.lock.return_value = asyncio.Lock()
        bot.history.get.return_value = []
        bot.history.add_exchange = mock.AsyncMock()
        bot.history.set_reply_ids = mock.AsyncMock()
        bot.presence_note.return_value = None
        bot.member_notes = mock.AsyncMock(return_value="")
        bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text="Réponse", web_used=False))
        self.channel = mock.Mock(id=CHANNEL_ID)

    @staticmethod
    async def read(message: mock.Mock, content: str) -> PendingMessage:
        line = {"id": message.id, "date": "07/11/2026 18:00", "author": "membre", "content": content}
        return PendingMessage(message, line, "", [])

    def message(self, message_id: int = 1) -> mock.Mock:
        return mock.Mock(id=message_id, channel=self.channel, attachments=[], content="salut", stickers=[],
                         created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))

    def saved(self) -> list[list[int]]:
        return json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else []

    async def interrupted_during_call(self, start) -> None:
        """Arrête le bot proprement pendant l'appel à Claude (ce que fait un déploiement)."""
        calling = asyncio.Event()

        async def endless(*_args, **_kwargs):
            calling.set()
            await asyncio.Event().wait()
        self.bot.claude.ask = endless
        task = asyncio.create_task(start)
        await calling.wait()
        self.bot._stopping = True  # ce que fait ChatBot.close
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class MainChannelTest(WaitingTestCase):
    async def test_kept_during_the_call(self):
        def check(*_args, **_kwargs):
            self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])
            return mock.Mock(text="Réponse", web_used=False)
        self.bot.claude.ask.side_effect = check
        await ChatBot.handle_main_channel(self.bot, self.message())
        self.bot.claude.ask.assert_awaited_once()

    async def test_forgotten_only_after_sending(self):
        def check(*_args, **_kwargs):
            self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])
        self.bot.send_answer.side_effect = check
        await ChatBot.handle_main_channel(self.bot, self.message())
        self.bot.send_answer.assert_awaited_once()
        self.assertEqual(self.saved(), [])

    async def test_kept_when_bot_stops_during_the_call(self):
        await self.interrupted_during_call(ChatBot.handle_main_channel(self.bot, self.message()))
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])

    async def test_queued_message_kept_when_bot_stops(self):
        # Le scénario de l'audit : une réponse en cours, et un autre message qui attend derrière.
        calling = asyncio.Event()

        async def endless(*_args, **_kwargs):
            calling.set()
            await asyncio.Event().wait()
        self.bot.claude.ask = endless
        first = asyncio.create_task(ChatBot.handle_main_channel(self.bot, self.message(1)))
        await calling.wait()
        second = asyncio.create_task(ChatBot.handle_main_channel(self.bot, self.message(2)))
        await asyncio.sleep(0.01)
        self.bot._stopping = True  # ce que fait ChatBot.close
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1], [CHANNEL_ID, 2]])

    async def test_kept_when_call_fails_because_bot_stops(self):
        # À l'arrêt, le client Claude est fermé : l'appel en cours échoue, ce n'est pas une vraie réponse.
        async def closing(*_args, **_kwargs):
            self.bot._stopping = True
            raise ClaudeError("Je n'arrive pas à joindre mon cerveau pour l'instant, réessaie plus tard.")
        self.bot.claude.ask = closing
        await ChatBot.handle_main_channel(self.bot, self.message())
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])

    async def test_close_marks_bot_as_stopping(self):
        bot = object.__new__(ChatBot)
        bot._stopping, bot.presence, bot._waiters = False, None, {}
        bot.claude = mock.Mock(close=mock.AsyncMock())
        with mock.patch("discord.ext.commands.Bot.close", mock.AsyncMock()):
            await bot.close()
        self.assertTrue(bot._stopping)

    async def test_forgotten_after_warning_author(self):
        self.bot.claude.ask.side_effect = ClaudeError("Erreur.")
        await ChatBot.handle_main_channel(self.bot, self.message())
        self.bot.safe_reply.assert_awaited_once()
        self.assertEqual(self.saved(), [])

    async def test_forgotten_after_unexpected_error_in_history(self):
        self.bot.history.add_exchange.side_effect = RuntimeError("panne")
        with self.assertRaises(RuntimeError):
            await ChatBot.handle_main_channel(self.bot, self.message())
        self.assertEqual(self.saved(), [])

    async def test_forgotten_when_nothing_to_read(self):
        self.bot.read_main_message = mock.AsyncMock(return_value=None)
        await ChatBot.handle_main_channel(self.bot, self.message())
        self.assertEqual(self.saved(), [])

    async def test_one_write_per_message_during_absence(self):
        # Chaque message déjà en file repasse par answer_later : le fichier ne change pas pour autant.
        self.bot.available.return_value = False
        self.bot.answer_later = ChatBot.answer_later.__get__(self.bot)
        self.bot._answer_when_possible = mock.AsyncMock()
        self.bot._save_waiting = mock.Mock(wraps=self.bot._save_waiting)
        for message_id in range(1, 21):
            await ChatBot.handle_main_channel(self.bot, self.message(message_id))
        await asyncio.sleep(0)  # laisse se terminer l'attente lancée par answer_later
        self.assertEqual(self.bot._save_waiting.call_count, 20)
        self.assertEqual(len(self.saved()), 20)

    async def test_replayed_message_written_again_when_deferred(self):
        # Repris pendant une absence, il n'est plus dans le fichier : il doit y revenir pour attendre le retour.
        self.bot._replayed.add((CHANNEL_ID, 1))
        self.bot.available.return_value = False
        self.bot.answer_later = ChatBot.answer_later.__get__(self.bot)
        self.bot._answer_when_possible = mock.AsyncMock()
        await ChatBot.handle_main_channel(self.bot, self.message(1))
        await asyncio.sleep(0)
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])

    async def test_refused_for_lack_of_room_not_kept(self):
        self.bot.held_files = mock.Mock()
        self.bot.held_files.reserve.return_value = False
        await ChatBot.handle_main_channel(self.bot, self.message())
        self.assertFalse(self.file.exists())


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class MentionTest(WaitingTestCase):
    def mention(self, message_id: int = 1) -> mock.Mock:
        message = self.message(message_id)
        message.guild.me.display_name = "Rika"
        return message

    async def test_kept_during_the_call_then_forgotten(self):
        def check(*_args, **_kwargs):
            self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])
        self.bot.send_answer.side_effect = check
        await ChatBot.handle_mention(self.bot, self.mention())
        self.bot.send_answer.assert_awaited_once()
        self.assertEqual(self.saved(), [])

    async def test_kept_when_bot_stops_during_the_call(self):
        await self.interrupted_during_call(ChatBot.handle_mention(self.bot, self.mention()))
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])

    async def test_forgotten_after_warning_author(self):
        self.bot.claude.ask.side_effect = ClaudeError("Erreur.")
        await ChatBot.handle_mention(self.bot, self.mention())
        self.bot.safe_reply.assert_awaited_once()
        self.assertEqual(self.saved(), [])


class SafeWritingTest(WaitingTestCase):
    """Le fichier est écrit à chaque message : un arrêt brutal en pleine écriture ne doit pas tout faire perdre."""

    def test_interrupted_write_keeps_previous_file(self):
        self.bot.remember_waiting(self.message(1))
        with mock.patch("bot.files.os.replace", side_effect=OSError("arrêt brutal")), \
                self.assertLogs("chatbot.discord", "WARNING"):
            self.bot.remember_waiting(self.message(2))
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])

    def test_no_forced_disk_flush(self):
        # L'écriture se fait dans la boucle du bot : un fsync le figerait à chaque message sur un volume lent.
        with mock.patch("bot.files.os.fsync") as fsync:
            self.bot.remember_waiting(self.message(1))
        fsync.assert_not_called()
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])

    def test_no_temporary_file_left(self):
        self.bot.remember_waiting(self.message(1))
        self.assertEqual([path.name for path in self.file.parent.iterdir()], [self.file.name])

    def test_other_files_still_flushed_by_default(self):
        with mock.patch("bot.files.os.fsync") as fsync:
            write_atomic(self.file.with_name("autre.json"), "{}")
        fsync.assert_called_once()


class ReplayOnlyOnceTest(WaitingTestCase):
    """Un message repris au démarrage n'a droit qu'à un essai : il ne peut pas faire tomber le bot en boucle."""

    def setUp(self):
        super().setUp()
        self.file.write_text(json.dumps([[CHANNEL_ID, 1]]), encoding="utf-8")
        ChatBot.load_waiting(self.bot)
        self.channel.fetch_message = mock.AsyncMock(side_effect=lambda message_id: self.message(message_id))
        self.bot.get_channel.return_value = self.channel

    async def test_replayed_message_leaves_the_file(self):
        def check(message):
            self.bot.remember_queued(message)
            # Si le bot tombe ici, ce message ne sera pas repris une seconde fois.
            self.assertEqual(self.saved(), [])
        self.bot.on_message = mock.AsyncMock(side_effect=check)
        await ChatBot.replay_waiting(self.bot)
        self.bot.on_message.assert_awaited_once()

    async def test_new_message_kept_during_replay(self):
        def check(message):
            self.bot.remember_queued(self.message(2))
        self.bot.on_message = mock.AsyncMock(side_effect=check)
        await ChatBot.replay_waiting(self.bot)
        self.assertEqual(self.saved(), [[CHANNEL_ID, 2]])

    async def test_replayed_message_waiting_for_return_stays_in_file(self):
        # Absent au redémarrage : le message attend le retour du bot, parfois plusieurs heures.
        def check(message):
            self.bot.remember_waiting(message)
        self.bot.on_message = mock.AsyncMock(side_effect=check)
        await ChatBot.replay_waiting(self.bot)
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1]])


if __name__ == "__main__":
    unittest.main()
