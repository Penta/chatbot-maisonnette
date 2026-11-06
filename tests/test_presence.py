"""Tests du statut Discord (nuit, durées, reprise après un redémarrage). Lancer depuis la racine du projet : python -m unittest"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

from bot.presence import Kind, PresenceManager, Status, format_duration
from bot.presence_sources import PresenceSources, Track

UTC = timezone.utc
PARIS = ZoneInfo("Europe/Paris")
# Tous les types de statut activés (la nuit n'est jamais tirée au sort : pas de poids).
WEIGHTS = {kind.value: 1 for kind in Kind if kind is not Kind.NIGHT}
T0 = datetime(2026, 11, 3, 9, 0, tzinfo=UTC)  # mardi 3 novembre 2026, 10 h à Paris


def frozen_at(moment: datetime):
    """Fige l'heure vue par bot/presence.py (datetime.now) à moment, quelle que soit l'heure du PC."""
    class Frozen(datetime):
        @classmethod
        def now(cls, tz):  # sans fuseau, l'heure dépendrait du PC : le module en donne toujours un
            return moment.astimezone(tz)
    return mock.patch("bot.presence.datetime", Frozen)


def make_manager(folder: Path, night=None, weights=None) -> PresenceManager:
    settings = SimpleNamespace(
        presence_weights=WEIGHTS if weights is None else weights,
        history_file=folder / "conversation_history.json",
        deezer_playlists=(), steam_api_key=None,
        presence_night=night, presence_speed=1.0, timezone=PARIS,
    )
    return PresenceManager(SimpleNamespace(), SimpleNamespace(), settings)


def paris(*args) -> datetime:
    return datetime(*args, tzinfo=PARIS)


class IsNightTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def is_night(self, night, hour: int, minute: int = 0) -> bool:
        return make_manager(Path(self._tmp.name), night=night).is_night(paris(2026, 11, 3, hour, minute))

    def test_no_night(self):
        for hour in range(24):
            self.assertFalse(self.is_night(None, hour))

    def test_night_within_the_day(self):
        self.assertFalse(self.is_night((1, 7), 0, 59))
        self.assertTrue(self.is_night((1, 7), 1))  # heure de début : déjà la nuit
        self.assertTrue(self.is_night((1, 7), 4))
        self.assertTrue(self.is_night((1, 7), 6, 59))
        self.assertFalse(self.is_night((1, 7), 7))  # heure de fin : déjà le jour
        self.assertFalse(self.is_night((1, 7), 12))

    def test_night_across_midnight(self):
        self.assertFalse(self.is_night((23, 6), 22, 59))
        self.assertTrue(self.is_night((23, 6), 23))
        self.assertTrue(self.is_night((23, 6), 0))
        self.assertTrue(self.is_night((23, 6), 5, 59))
        self.assertFalse(self.is_night((23, 6), 6))
        self.assertFalse(self.is_night((23, 6), 12))


class NextHourTest(unittest.TestCase):
    def test_later_the_same_day(self):
        result = PresenceManager.next_hour(paris(2026, 11, 3, 10, 0), 14)
        self.assertEqual(result, datetime(2026, 11, 3, 13, 0, tzinfo=UTC))
        self.assertIs(result.tzinfo, UTC)

    def test_hour_already_passed(self):
        result = PresenceManager.next_hour(paris(2026, 11, 3, 10, 0), 6)
        self.assertEqual(result, datetime(2026, 11, 4, 5, 0, tzinfo=UTC))
        self.assertIs(result.tzinfo, UTC)

    def test_exact_hour_is_the_next_day(self):
        self.assertEqual(PresenceManager.next_hour(paris(2026, 11, 3, 6, 0), 6),
                         datetime(2026, 11, 4, 5, 0, tzinfo=UTC))

    def test_just_before_the_hour(self):
        self.assertEqual(PresenceManager.next_hour(paris(2026, 11, 3, 5, 59, 59), 6),
                         datetime(2026, 11, 3, 5, 0, tzinfo=UTC))

    def test_spring_forward(self):
        # Nuit du 28 au 29 mars 2026 : 2 h devient 3 h, 6 h du matin est à 4 h UTC (et non 5 h).
        self.assertEqual(PresenceManager.next_hour(paris(2026, 3, 28, 12, 0), 6),
                         datetime(2026, 3, 29, 4, 0, tzinfo=UTC))
        self.assertEqual(PresenceManager.next_hour(paris(2026, 3, 29, 1, 30), 6),
                         datetime(2026, 3, 29, 4, 0, tzinfo=UTC))

    def test_fall_back(self):
        # Nuit du 24 au 25 octobre 2026 : 3 h redevient 2 h, 6 h du matin est à 5 h UTC (et non 4 h).
        self.assertEqual(PresenceManager.next_hour(paris(2026, 10, 24, 12, 0), 6),
                         datetime(2026, 10, 25, 5, 0, tzinfo=UTC))
        self.assertEqual(PresenceManager.next_hour(paris(2026, 10, 25, 1, 0), 6),
                         datetime(2026, 10, 25, 5, 0, tzinfo=UTC))


class FormatDurationTest(unittest.TestCase):
    def test_less_than_a_minute(self):
        self.assertEqual(format_duration(0), "1 min")
        self.assertEqual(format_duration(20), "1 min")

    def test_minutes(self):
        self.assertEqual(format_duration(60), "1 min")
        self.assertEqual(format_duration(25 * 60), "25 min")
        self.assertEqual(format_duration(59 * 60), "59 min")

    def test_whole_hours(self):
        self.assertEqual(format_duration(3600), "1 h")
        self.assertEqual(format_duration(2 * 3600), "2 h")

    def test_hours_and_minutes(self):
        self.assertEqual(format_duration(3600 + 5 * 60), "1 h 05")
        self.assertEqual(format_duration(2 * 3600 + 30 * 60), "2 h 30")


class StatusDictTest(unittest.TestCase):
    def assert_round_trip(self, status: Status) -> None:
        data = json.loads(json.dumps(status.to_dict()))  # tel qu'écrit dans presence_state.json
        self.assertEqual(Status.from_dict(data), status)

    def test_every_kind(self):
        for kind in Kind:
            with self.subTest(kind=kind):
                self.assert_round_trip(Status(kind))
                self.assert_round_trip(Status(kind, title="Un titre"))

    def test_anime_episode(self):
        self.assert_round_trip(Status(Kind.ANIME, title="Frieren", episode=7))

    def test_music_track(self):
        self.assert_round_trip(Status(Kind.MUSIC, track=Track("Clair de lune", "Debussy", 300)))

    def test_game_details(self):
        self.assert_round_trip(Status(Kind.GAME, title="Celeste", details="Un jeu de plateforme."))

    def test_empty_fields_left_out(self):
        self.assertEqual(Status(Kind.ONLINE).to_dict(), {"kind": "en_ligne"})


class GameDetailsTest(unittest.IsolatedAsyncioTestCase):
    async def details(self, **fields):
        info = {"type": "game", "name": "Celeste", **fields}
        response = {"42": {"success": True, "data": info}}
        with mock.patch("bot.presence_sources.get_json", mock.AsyncMock(return_value=response)):
            return await PresenceSources.game_details(42)

    async def test_unreleased_game_rejected(self):
        self.assertIsNone(await self.details(release_date={"coming_soon": True, "date": "2027"}))

    async def test_released_game_kept(self):
        game = await self.details(release_date={"coming_soon": False, "date": "25 janv. 2018"})
        self.assertEqual(game.name, "Celeste")

    async def test_missing_release_date_counts_as_released(self):
        self.assertEqual((await self.details()).name, "Celeste")


class RestoreTest(unittest.IsolatedAsyncioTestCase):
    """presence_state.json relu au démarrage : le statut reprend là où il en serait."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.state_file = self.folder / "presence_state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, status: Status, minutes: float, program=(), saved_at=T0, **options) -> None:
        """Enregistre status (pour minutes à partir de saved_at, puis program) comme le ferait le bot."""
        manager = make_manager(self.folder, **options)
        manager.status = status
        manager.step_end = saved_at + timedelta(minutes=minutes)
        manager.program = [(step, seconds) for step, seconds in program]
        manager.status_end = manager.step_end + timedelta(seconds=sum(seconds for _, seconds in program))
        with frozen_at(saved_at):
            manager.save()

    def restore(self, now: datetime, **options) -> PresenceManager:
        manager = make_manager(self.folder, **options)
        with frozen_at(now):
            manager.restore()
        return manager

    def assert_fresh_start(self, manager: PresenceManager) -> None:
        """Rien de repris : la boucle tirera un nouveau statut à sa première étape."""
        self.assertEqual(manager.status, Status(Kind.ONLINE))
        self.assertEqual(manager.program, [])
        self.assertEqual(manager.step_end, datetime.min.replace(tzinfo=UTC))
        self.assertIsNone(manager.status_end)

    def episodes(self, *numbers: int) -> list:
        return [(Status(Kind.ANIME, title="Frieren", episode=n), 24 * 60) for n in numbers]

    def test_anime_moves_on_by_the_episodes_missed(self):
        # Épisode 3 jusqu'à +10 min, puis 4 (+34), 5 (+58) et 6 (+82) : à +40, c'est l'épisode 5.
        self.save(self.episodes(3)[0][0], 10, self.episodes(4, 5, 6))
        manager = self.restore(T0 + timedelta(minutes=40))
        self.assertEqual(manager.status, Status(Kind.ANIME, title="Frieren", episode=5))
        self.assertEqual(manager.program, self.episodes(6))
        self.assertEqual(manager.step_end, T0 + timedelta(minutes=58))
        self.assertEqual(manager.status_end, T0 + timedelta(minutes=82))

    def test_short_restart_keeps_the_current_step(self):
        self.save(self.episodes(3)[0][0], 10, self.episodes(4, 5))
        manager = self.restore(T0 + timedelta(minutes=5))
        self.assertEqual(manager.status.episode, 3)
        self.assertEqual(manager.program, self.episodes(4, 5))
        self.assertEqual(manager.step_end, T0 + timedelta(minutes=10))

    def test_game_details_restored(self):
        game = Status(Kind.GAME, title="Celeste", details="Un jeu de plateforme.")
        self.save(game, 60)
        self.assertEqual(self.restore(T0 + timedelta(minutes=30)).status, game)

    def test_file_written_by_the_bot(self):
        # Format du fichier tel qu'enregistré en production (musique : un morceau par étape).
        track = {"title": "Clair de lune", "artist": "Debussy", "duration": 300}
        self.state_file.write_text(json.dumps({
            "saved_at": "2026-11-03T09:00:00+00:00",
            "status": {"kind": "musique", "track": track},
            "step_end": "2026-11-03T09:02:00+00:00",
            "status_end": "2026-11-03T09:12:00+00:00",
            "program": [[{"kind": "musique", "track": {**track, "title": "Rêverie"}}, 300.0],
                        [{"kind": "musique", "track": {**track, "title": "Arabesque"}}, 300.0]],
            "recent_custom": ["Je range ma chambre"],
            "recent_kinds": ["pensee", "musique"],
        }), encoding="utf-8")
        manager = self.restore(T0 + timedelta(minutes=4))
        self.assertEqual(manager.status, Status(Kind.MUSIC, track=Track("Rêverie", "Debussy", 300)))
        self.assertEqual([step.track.title for step, _ in manager.program], ["Arabesque"])
        self.assertEqual(manager.step_end, T0 + timedelta(minutes=7))
        self.assertEqual(list(manager.recent_custom), ["Je range ma chambre"])
        self.assertEqual(list(manager.recent_kinds), [Kind.THOUGHT, Kind.MUSIC])

    def test_finished_during_the_restart(self):
        self.save(self.episodes(3)[0][0], 10, self.episodes(4, 5))
        self.assert_fresh_start(self.restore(T0 + timedelta(hours=2)))

    def test_finished_exactly_at_restart(self):
        self.save(self.episodes(3)[0][0], 10, self.episodes(4))
        self.assert_fresh_start(self.restore(T0 + timedelta(minutes=34)))

    def test_recent_lists_kept_even_when_the_status_is_over(self):
        manager = make_manager(self.folder)
        manager.status = Status(Kind.THOUGHT, title="Je range ma chambre")
        manager.step_end = T0 + timedelta(minutes=10)
        manager.recent_custom.append("Je range ma chambre")
        manager.recent_kinds.extend([Kind.MUSIC, Kind.THOUGHT])
        with frozen_at(T0):
            manager.save()
        restored = self.restore(T0 + timedelta(hours=1))
        self.assert_fresh_start(restored)
        self.assertEqual(list(restored.recent_custom), ["Je range ma chambre"])
        self.assertEqual(list(restored.recent_kinds), [Kind.MUSIC, Kind.THOUGHT])

    def test_cut_by_the_night(self):
        # Partie commencée à 1 h (Paris) pour 6 h : la nuit de 2 h à 6 h l'a interrompue.
        saved_at = paris(2026, 11, 3, 1, 0).astimezone(UTC)
        game = Status(Kind.GAME, title="Celeste")
        self.save(game, 6 * 60, saved_at=saved_at, night=(2, 6))
        self.assert_fresh_start(self.restore(paris(2026, 11, 3, 6, 10), night=(2, 6)))
        # Sans nuit, ou redémarré avant 2 h, la partie continue.
        self.assertEqual(self.restore(paris(2026, 11, 3, 6, 10), night=None).status, game)
        self.assertEqual(self.restore(paris(2026, 11, 3, 1, 30), night=(2, 6)).status, game)

    def test_night_saved(self):
        self.save(Status(Kind.NIGHT), 60)
        self.assert_fresh_start(self.restore(T0 + timedelta(minutes=10)))

    def test_kind_disabled_since(self):
        self.save(Status(Kind.GAME, title="Celeste"), 60)
        self.assert_fresh_start(self.restore(T0 + timedelta(minutes=10), weights={**WEIGHTS, "jeu": 0}))
        without_games = {name: weight for name, weight in WEIGHTS.items() if name != "jeu"}
        self.assert_fresh_start(self.restore(T0 + timedelta(minutes=10), weights=without_games))

    def test_missing_file(self):
        with self.assertNoLogs("chatbot.presence", "WARNING"):
            manager = self.restore(T0)
        self.assert_fresh_start(manager)

    def test_unreadable_file(self):
        self.save(Status(Kind.GAME, title="Celeste"), 60)
        valid = json.loads(self.state_file.read_text(encoding="utf-8"))
        contents = {
            "vide": "",
            "tronqué": json.dumps(valid)[:-15],
            "liste": "[]",
            "sans statut": json.dumps({**valid, "status": None}),
            "type inconnu": json.dumps({**valid, "status": {"kind": "volant"}}),
            "morceau incomplet": json.dumps({**valid, "status": {"kind": "musique", "track": {"title": "x"}}}),
            "date illisible": json.dumps({**valid, "step_end": "demain"}),
            "date en nombre": json.dumps({**valid, "saved_at": 12}),
            "durée illisible": json.dumps({**valid, "program": [[{"kind": "jeu"}, "longtemps"]]}),
            "programme mal formé": json.dumps({**valid, "program": 5}),
            "statuts récents mal formés": json.dumps({**valid, "recent_custom": 5}),
        }
        for name, text in contents.items():
            with self.subTest(name):
                self.state_file.write_text(text, encoding="utf-8")
                with self.assertLogs("chatbot.presence", "WARNING"):
                    manager = self.restore(T0 + timedelta(minutes=10))
                self.assertEqual(manager.status, Status(Kind.ONLINE))
                self.assertEqual(manager.program, [])

    def test_date_without_timezone(self):
        self.save(Status(Kind.GAME, title="Celeste"), 60)
        valid = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.state_file.write_text(json.dumps({**valid, "step_end": "2026-11-03T10:00:00"}), encoding="utf-8")
        with self.assertLogs("chatbot.presence", "WARNING"):
            manager = self.restore(T0 + timedelta(minutes=10))
        self.assertEqual(manager.status, Status(Kind.ONLINE))

    async def step(self, manager: PresenceManager, now: datetime) -> tuple[mock.AsyncMock, mock.AsyncMock]:
        """Une étape de la boucle, sans tirage au sort ni Discord : renvoie (draw, set_status)."""
        draw = mock.AsyncMock(return_value=[(Status(Kind.ONLINE), 600)])
        set_status = mock.AsyncMock()
        with mock.patch.object(manager, "draw", draw), mock.patch.object(manager, "set_status", set_status), \
                frozen_at(now):
            await manager._step()
        return draw, set_status

    async def test_loop_goes_on_with_the_restored_program(self):
        self.save(self.episodes(3)[0][0], 10, self.episodes(4, 5))
        manager = self.restore(T0 + timedelta(minutes=20))  # épisode 4 jusqu'à +34
        draw, set_status = await self.step(manager, T0 + timedelta(minutes=20))
        draw.assert_not_called()
        set_status.assert_not_called()
        draw, set_status = await self.step(manager, T0 + timedelta(minutes=35))
        draw.assert_not_called()
        set_status.assert_awaited_once_with(Status(Kind.ANIME, title="Frieren", episode=5))

    async def test_loop_draws_after_a_finished_status(self):
        self.save(self.episodes(3)[0][0], 10)
        manager = self.restore(T0 + timedelta(hours=1))
        draw, set_status = await self.step(manager, T0 + timedelta(hours=1))
        draw.assert_awaited_once()
        set_status.assert_awaited_once_with(Status(Kind.ONLINE))


if __name__ == "__main__":
    unittest.main()
