"""Tests des messages spontanés (fin d'un anime ou d'une partie de jeu, salon calme). Lancer depuis la racine du
projet : python -m unittest"""

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from discord.utils import time_snowflake

from bot.discord_bot import ChatBot
from bot.presence import Kind, Status
from bot.presence_sources import Track
from bot.spontaneous import SpontaneousLog, finished_activity, is_silence, occasion_note

UTC = timezone.utc
NOW = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
CHANNEL_ID = 10
ANIME_EP4 = Status(Kind.ANIME, title="Frieren", episode=4)
MUSIC = Status(Kind.MUSIC, track=Track(title="Idol", artist="YOASOBI", duration=200))


def message_id(when: datetime) -> int:
    """Identifiant d'un message Discord écrit à cette date."""
    return time_snowflake(when)


class FinishedActivityTest(unittest.TestCase):
    def test_end_of_anime_or_game(self):
        self.assertTrue(finished_activity(ANIME_EP4, MUSIC))
        self.assertTrue(finished_activity(ANIME_EP4, Status(Kind.ANIME, title="Dandadan", episode=1)))
        self.assertTrue(finished_activity(Status(Kind.GAME, title="Hades"), Status(Kind.ONLINE)))

    def test_not_an_end(self):
        # Épisode suivant du même anime : la séance continue.
        self.assertFalse(finished_activity(ANIME_EP4, Status(Kind.ANIME, title="Frieren", episode=5)))
        # Part se coucher ou s'absenter : il ne lance pas la conversation.
        for after in (Status(Kind.NIGHT), Status(Kind.DND), Status(Kind.IDLE, title="courses")):
            self.assertFalse(finished_activity(ANIME_EP4, after))
        # Musique, pensée, en ligne : rien de fini à raconter.
        self.assertFalse(finished_activity(MUSIC, Status(Kind.ONLINE)))
        self.assertFalse(finished_activity(Status(Kind.THOUGHT, title="…"), Status(Kind.ONLINE)))


class SilenceTest(unittest.TestCase):
    def test_silence(self):
        for text in ("RIEN", "RIEN.", " rien\n", "**Rien**", ""):
            self.assertTrue(is_silence(text), text)

    def test_real_message(self):
        self.assertFalse(is_silence("Rien à dire de plus, cet épisode était parfait."))

    def test_occasion_note_names_what_was_finished(self):
        note = occasion_note(ANIME_EP4, datetime(2026, 11, 7, 16, 42, tzinfo=ZoneInfo("Europe/Paris")))
        self.assertIn("16:42", note)
        self.assertIn("l'épisode 4 de l'anime « Frieren »", note)
        self.assertIn("RIEN", note)


class SpontaneousLogTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "spontaneous_messages.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_when_the_channel_is_calm(self):
        log = SpontaneousLog(self.path, per_day=1)
        self.assertFalse(log.allowed(NOW, message_id(NOW - timedelta(hours=2))))
        self.assertTrue(log.allowed(NOW, message_id(NOW - timedelta(hours=3))))
        self.assertTrue(log.allowed(NOW, None))  # salon vide

    def test_calm_shortened_by_presence_speed(self):
        log = SpontaneousLog(self.path, per_day=1, speed=10)
        self.assertTrue(log.allowed(NOW, message_id(NOW - timedelta(minutes=18))))
        self.assertFalse(log.allowed(NOW, message_id(NOW - timedelta(minutes=17))))

    def test_limit_per_day(self):
        log = SpontaneousLog(self.path, per_day=2)
        old = message_id(NOW - timedelta(hours=10))
        log.record(1, NOW - timedelta(hours=20))
        log.record(2, NOW - timedelta(hours=12))
        self.assertFalse(log.allowed(NOW, old))
        # 24 h après le premier, il compte de nouveau.
        self.assertTrue(log.allowed(NOW + timedelta(hours=4, minutes=1), old))

    def test_not_twice_without_an_answer(self):
        # Aucun membre n'a écrit depuis son dernier message spontané, 6 h plus tard : il ne reparle pas seul.
        log = SpontaneousLog(self.path, per_day=5)
        log.record(message_id(NOW - timedelta(hours=6)), NOW - timedelta(hours=6))
        self.assertFalse(log.allowed(NOW, message_id(NOW - timedelta(hours=8))))
        self.assertFalse(log.allowed(NOW, None))  # plus aucun message de membre parmi les derniers du salon
        # Un membre a écrit depuis (il y a 4 h) : le salon est de nouveau calme, il peut reparler.
        self.assertTrue(log.allowed(NOW, message_id(NOW - timedelta(hours=4))))

    def test_kept_after_a_restart(self):
        SpontaneousLog(self.path, per_day=1).record(message_id(NOW - timedelta(hours=6)), NOW - timedelta(hours=6))
        log = SpontaneousLog(self.path, per_day=5)
        # Pas deux fois de suite, même après un redémarrage.
        self.assertFalse(log.allowed(NOW, message_id(NOW - timedelta(hours=8))))
        self.assertFalse(SpontaneousLog(self.path, per_day=1).allowed(NOW, None))  # limite du jour atteinte

    def test_unreadable_file_ignored(self):
        self.path.write_text("pas du json", encoding="utf-8")
        with self.assertLogs("chatbot.spontaneous", "WARNING"):
            log = SpontaneousLog(self.path, per_day=1)
        self.assertTrue(log.allowed(NOW, None))


@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
@mock.patch("bot.discord_bot.discord.utils.utcnow", lambda: NOW)
@mock.patch("bot.discord_bot.random.random", lambda: 0.0)
class SpeakUpTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.after = MUSIC
        self.bot = bot = mock.Mock()
        bot.settings.timezone = ZoneInfo("Europe/Paris")
        bot.settings.channel_id = CHANNEL_ID
        bot.spontaneous = SpontaneousLog(Path(self._tmp.name) / "spontaneous_messages.json", per_day=1)
        bot.presence.status = self.after
        bot._pending, bot._preparing, bot._waiters = {}, {}, {}
        bot.history.lock.return_value = asyncio.Lock()
        bot.history.get.return_value = []
        bot.history.add_exchange = mock.AsyncMock()
        bot.presence_note.return_value = "[ton statut Discord : tu écoutes « Idol » de YOASOBI]"
        bot.memory.notes_for.return_value = "[fiches]"
        bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text="L'épisode 4 de Frieren, quelle claque.",
                                                                web_used=False))
        self.channel = mock.Mock(id=CHANNEL_ID, guild=None)
        self.channel.send = mock.AsyncMock(return_value=mock.Mock(id=message_id(NOW)))
        bot.get_channel.return_value = self.channel
        self.channel_messages((message_id(NOW - timedelta(hours=5)), False))  # un membre a écrit il y a 5 h

    def tearDown(self):
        self._tmp.cleanup()

    def channel_messages(self, *messages: tuple[int, bool], author: int | None = None) -> None:
        """Derniers messages du salon, du plus récent au plus ancien : (identifiant, écrit par un bot). Chacun a
        son auteur, sauf author : le même pour tous."""
        found = [mock.Mock(id=i, author=mock.Mock(id=author or 1000 + n, bot=is_bot))
                 for n, (i, is_bot) in enumerate(messages)]

        async def history(limit):
            for message in found[:limit]:
                yield message
        self.channel.history = history

    async def speak_up(self, before=ANIME_EP4):
        await ChatBot.speak_up(self.bot, before, self.after)

    async def test_speaks_after_the_last_episode(self):
        await self.speak_up()
        self.channel.send.assert_awaited_once_with("L'épisode 4 de Frieren, quelle claque.")
        parts = [text for text, _ in self.bot.claude.ask.await_args.args[1]]
        self.assertIn("l'épisode 4 de l'anime « Frieren »", parts[-1])
        self.bot.history.add_exchange.assert_awaited_once()
        self.bot.memory.notes_for.assert_called_once_with([1000])  # fiche du membre qui a écrit il y a 5 h
        self.assertEqual(self.bot.claude.ask.await_args.kwargs["member_notes"], "[fiches]")
        # Personne ne lui a répondu : à la fin de l'anime suivant, il se tait.
        self.channel_messages((message_id(NOW), True), (message_id(NOW - timedelta(hours=5)), False))
        self.bot.spontaneous.per_day = 5
        await self.speak_up()
        self.assertEqual(self.channel.send.await_count, 1)

    async def test_messages_of_bots_are_not_an_answer(self):
        # À 14 h, Rika parle seule de son anime. À 15 h, elle redémarre et poste « En ligne » ; un autre bot écrit
        # à 16 h. À 19 h, personne ne lui a répondu : elle ne reparle pas seule.
        own = message_id(NOW - timedelta(hours=5))
        self.bot.spontaneous.per_day = 5
        self.bot.spontaneous.record(own, NOW - timedelta(hours=5))
        self.channel_messages(
            (message_id(NOW - timedelta(hours=3)), True),  # autre bot
            (message_id(NOW - timedelta(hours=4)), True),  # « En ligne »
            (own, True),
            (message_id(NOW - timedelta(hours=8)), False),  # dernier message d'un membre
        )
        await self.speak_up()
        self.bot.claude.ask.assert_not_awaited()

    async def test_messages_of_bots_do_not_break_the_calm(self):
        # Un membre a écrit il y a 5 h ; un autre bot écrit depuis toutes les heures : le salon reste calme.
        self.channel_messages(
            (message_id(NOW - timedelta(minutes=10)), True),
            (message_id(NOW - timedelta(hours=1)), True),
            (message_id(NOW - timedelta(hours=5)), False),
        )
        await self.speak_up()
        self.channel.send.assert_awaited_once()

    async def test_notes_of_recent_members_once_each(self):
        self.channel_messages((message_id(NOW - timedelta(hours=4)), False), (message_id(NOW - timedelta(hours=5)), True),
                              (message_id(NOW - timedelta(hours=6)), False), author=7)
        await self.speak_up()
        self.bot.memory.notes_for.assert_called_once_with([7])  # sans les bots, et une seule fois chacun

    async def test_quiet_when_discord_does_not_answer(self):
        async def history(limit):
            raise OSError("réseau coupé")
            yield
        self.channel.history = history
        with self.assertLogs("chatbot.discord", "WARNING"):
            await self.speak_up()
        self.bot.claude.ask.assert_not_awaited()

    async def test_silence_not_sent(self):
        self.bot.claude.ask.return_value = mock.Mock(text="RIEN", web_used=False)
        await self.speak_up()
        self.channel.send.assert_not_awaited()
        self.bot.history.add_exchange.assert_not_awaited()
        self.assertEqual(self.bot.spontaneous.sent, [])  # ne compte pas dans la limite du jour

    async def test_quiet_when_the_channel_is_busy(self):
        self.channel_messages((message_id(NOW - timedelta(minutes=30)), False))
        await self.speak_up()
        self.bot._pending[CHANNEL_ID] = [mock.Mock()]
        self.channel_messages((message_id(NOW - timedelta(hours=5)), False))
        await self.speak_up()
        self.bot.claude.ask.assert_not_awaited()

    async def test_quiet_when_nothing_was_finished(self):
        # Épisode 3 fini, l'épisode 4 commence : la séance continue.
        self.after = self.bot.presence.status = ANIME_EP4
        await self.speak_up(before=Status(Kind.ANIME, title="Frieren", episode=3))
        # Fin d'une playlist de musique.
        self.after = self.bot.presence.status = Status(Kind.ONLINE)
        await self.speak_up(before=MUSIC)
        self.bot.claude.ask.assert_not_awaited()

    async def test_quiet_after_a_skip(self):
        # Un administrateur passe le statut avec /skip en plein anime : Rika ne l'a pas fini, elle n'en parle pas.
        self.bot.speak_up = lambda before, after: ChatBot.speak_up(self.bot, before, after)
        await ChatBot.on_status_change(self.bot, ANIME_EP4, self.after, True)
        self.bot.claude.ask.assert_not_awaited()
        await ChatBot.on_status_change(self.bot, ANIME_EP4, self.after, False)  # fin normale de l'anime
        self.bot.claude.ask.assert_awaited_once()

    async def test_not_every_time(self):
        with mock.patch("bot.discord_bot.random.random", lambda: 0.5):
            await self.speak_up()
        self.bot.claude.ask.assert_not_awaited()

    async def test_disabled(self):
        self.bot.spontaneous = None
        await self.speak_up()
        self.bot.claude.ask.assert_not_awaited()
