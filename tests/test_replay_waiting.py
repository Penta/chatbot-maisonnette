"""Tests de la reprise au démarrage des messages restés sans réponse (waiting_messages.json).
Lancer depuis la racine du projet : python -m unittest"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.discord_bot import ChatBot

CHANNEL_ID = 10


def fake_bot(waiting_file: Path) -> mock.Mock:
    """Faux bot, avec le vrai code de gestion du fichier d'attente."""
    bot = mock.Mock()
    bot._waiting_file = waiting_file
    bot._waiting, bot._replayed, bot._to_replay = set(), set(), set()
    bot._save_waiting = lambda: ChatBot._save_waiting(bot)
    bot.remember_waiting = lambda message: ChatBot.remember_waiting(bot, message)
    bot.on_message = mock.AsyncMock()
    return bot


class ReplayWaitingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.file = Path(folder.name) / "waiting_messages.json"
        self.file.write_text(json.dumps([[CHANNEL_ID, 1], [CHANNEL_ID, 2]]), encoding="utf-8")
        self.bot = fake_bot(self.file)
        ChatBot.load_waiting(self.bot)  # fait par setup_hook, avant la connexion à Discord
        self.channel = mock.Mock(id=CHANNEL_ID)
        self.channel.fetch_message = mock.AsyncMock(side_effect=lambda message_id: mock.Mock(id=message_id))
        self.bot.get_channel.return_value = self.channel

    def saved(self) -> list[list[int]]:
        return json.loads(self.file.read_text(encoding="utf-8"))

    def new_message_during_download(self) -> None:
        """Un membre écrit pendant le téléchargement du premier message : il est mis en attente."""
        async def fetch(message_id: int) -> mock.Mock:
            if message_id == 1:
                self.bot.remember_waiting(mock.Mock(id=3, channel=self.channel))
            return mock.Mock(id=message_id)
        self.channel.fetch_message.side_effect = fetch

    async def test_message_arrived_during_download_stays_in_file(self):
        self.new_message_during_download()
        await ChatBot.replay_waiting(self.bot)
        self.assertEqual(self.bot._waiting, {(CHANNEL_ID, 3)})
        self.assertEqual(self.saved(), [[CHANNEL_ID, 3]])
        self.assertEqual([call.args[0].id for call in self.bot.on_message.await_args_list], [1, 2])

    async def test_new_message_does_not_erase_saved_ones_from_file(self):
        # Le bot peut être arrêté en plein téléchargement : le fichier doit encore tout contenir.
        seen = []
        self.new_message_during_download()
        download = self.channel.fetch_message.side_effect

        async def fetch(message_id: int) -> mock.Mock:
            message = await download(message_id)
            seen.append(self.saved())
            return message
        self.channel.fetch_message.side_effect = fetch
        await ChatBot.replay_waiting(self.bot)
        self.assertEqual(seen[0], [[CHANNEL_ID, 1], [CHANNEL_ID, 2], [CHANNEL_ID, 3]])

    async def test_message_received_before_replay_keeps_the_others(self):
        # Paul écrit pendant que le bot se connecte, avant on_ready : A et B restent dans le fichier, et son
        # message n'est pas repris, puisqu'il est déjà en cours de traitement.
        self.bot.remember_waiting(mock.Mock(id=3, channel=self.channel))
        self.assertEqual(self.saved(), [[CHANNEL_ID, 1], [CHANNEL_ID, 2], [CHANNEL_ID, 3]])
        await ChatBot.replay_waiting(self.bot)
        self.assertEqual([call.args[0].id for call in self.bot.on_message.await_args_list], [1, 2])
        self.assertEqual(self.saved(), [[CHANNEL_ID, 3]])

    async def test_replay_only_once(self):
        # on_ready ne lance qu'une reprise, mais rien ne doit dépendre d'une seconde lecture du fichier.
        await ChatBot.replay_waiting(self.bot)
        self.bot.on_message.reset_mock()
        await ChatBot.replay_waiting(self.bot)
        self.bot.on_message.assert_not_awaited()

    def test_missing_file_ignored(self):
        self.file.unlink()
        bot = fake_bot(self.file)
        ChatBot.load_waiting(bot)
        self.assertEqual(bot._waiting, set())

    def test_unreadable_file_ignored(self):
        for content in ("pas du json", "[1, 2]", "[[1, 2, 3]]", "{\"a\": 1}"):
            with self.subTest(content):
                self.file.write_text(content, encoding="utf-8")
                bot = fake_bot(self.file)
                with self.assertLogs("chatbot.discord", "WARNING"):
                    ChatBot.load_waiting(bot)
                self.assertEqual(bot._waiting, set())

    async def test_replay_error_is_logged_and_others_still_replayed(self):
        self.bot.on_message.side_effect = [RuntimeError("panne"), None]
        with self.assertLogs("chatbot", level="ERROR") as logs:
            await ChatBot.replay_waiting(self.bot)
        self.assertIn("panne", logs.output[0] + str(logs.records[0].exc_info))
        self.assertEqual(self.bot.on_message.await_count, 2)
