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
from bot.presence_sources import DEEZER_TRACKS_URL, Anime, PresenceSources, SourceError, Track

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
        deezer_playlists=(),
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


class FetchAnimeTest(unittest.IsolatedAsyncioTestCase):
    """Seuls les épisodes déjà diffusés sont proposés, même quand AniList connaît le total prévu."""

    async def fetch(self, **fields) -> list:
        media = {"title": {"romaji": "Frieren"}, "duration": 24, **fields}
        response = {"data": {"Page": {"media": [media]}}}
        with mock.patch("bot.presence_sources.get_json", mock.AsyncMock(return_value=response)):
            return await PresenceSources.fetch_anime(None)

    async def test_finished_anime_keeps_all_its_episodes(self):
        anime = await self.fetch(status="FINISHED", episodes=28)
        self.assertEqual([a.episodes for a in anime], [28])

    async def test_airing_anime_counts_aired_episodes_even_with_known_total(self):
        anime = await self.fetch(status="RELEASING", episodes=12, nextAiringEpisode={"episode": 7})
        self.assertEqual([a.episodes for a in anime], [6])

    async def test_airing_anime_without_known_total(self):
        anime = await self.fetch(status="RELEASING", episodes=None, nextAiringEpisode={"episode": 7})
        self.assertEqual([a.episodes for a in anime], [6])

    async def test_unreleased_anime_left_out(self):
        self.assertEqual(await self.fetch(status="NOT_YET_RELEASED", episodes=12), [])
        self.assertEqual(await self.fetch(status="NOT_YET_RELEASED", episodes=12,
                                          nextAiringEpisode={"episode": 1}), [])

    async def test_movie_recognised_by_its_format(self):
        anime = await self.fetch(status="FINISHED", format="MOVIE", episodes=1)
        self.assertEqual([a.movie for a in anime], [True])
        anime = await self.fetch(status="FINISHED", format="MOVIE", episodes=3)  # film en trois parties
        self.assertEqual([a.movie for a in anime], [True])
        anime = await self.fetch(status="FINISHED", format="SPECIAL", episodes=1)  # épisode spécial unique
        self.assertEqual([a.movie for a in anime], [True])
        anime = await self.fetch(status="RELEASING", format="TV", episodes=12, nextAiringEpisode={"episode": 2})
        self.assertEqual([(a.episodes, a.movie) for a in anime], [(1, False)])

    async def test_airing_anime_with_unknown_aired_count_left_out(self):
        # En pause ou date du prochain épisode inconnue : impossible de savoir lesquels sont sortis.
        self.assertEqual(await self.fetch(status="RELEASING", episodes=12), [])
        self.assertEqual(await self.fetch(status="HIATUS", episodes=24), [])


class DrawAnimeTest(unittest.IsolatedAsyncioTestCase):
    """Un film s'affiche sans numéro d'épisode ; une série qui n'a qu'un épisode sorti garde le sien."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.manager = make_manager(Path(self._tmp.name), weights={Kind.ANIME.value: 1})

    def tearDown(self):
        self._tmp.cleanup()

    async def draw(self, anime: Anime) -> list:
        self.manager.sources.anime = [anime]
        return [status for status, _ in await self.manager.draw()]

    async def test_movie_has_no_episode_number(self):
        self.assertEqual(await self.draw(Anime("Kimi no Na wa.", 1, 106, movie=True)),
                         [Status(Kind.ANIME, title="Kimi no Na wa.")])

    async def test_movie_in_several_parts_lasts_all_its_parts(self):
        self.manager.sources.anime = [Anime("Byousoku 5 Centimeter", 3, 21, movie=True)]
        program = await self.manager.draw()
        self.assertEqual(program, [(Status(Kind.ANIME, title="Byousoku 5 Centimeter"), 3 * 21 * 60)])

    async def test_series_with_one_aired_episode_keeps_its_number(self):
        self.assertEqual(await self.draw(Anime("Nouvelle saison", 1, 24)),
                         [Status(Kind.ANIME, title="Nouvelle saison", episode=1)])


class SourcesCacheTest(unittest.TestCase):
    def test_old_cache_without_movie_field(self):
        # Fichier écrit avant que le format soit lu : un seul épisode y voulait dire « film ».
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "presence_sources.json"
            cache.write_text(json.dumps({"anime": [{"title": "Film", "episodes": 1, "duration": 100},
                                                   {"title": "Série", "episodes": 12, "duration": 24}]}),
                             encoding="utf-8")
            sources = PresenceSources(cache, ())
            sources.load_cache()
        self.assertEqual(sources.anime, [Anime("Film", 1, 100, movie=True), Anime("Série", 12, 24)])

    def test_damaged_anime_entry_ignored_with_a_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "presence_sources.json"
            cache.write_text(json.dumps({"anime": ["Frieren"]}), encoding="utf-8")
            sources = PresenceSources(cache, ())
            with self.assertLogs("chatbot.presence", "WARNING"):
                sources.load_cache()
        self.assertEqual(sources.anime, [])

    def test_old_list_of_the_whole_steam_store_ignored(self):
        # Avant : toute la boutique Steam, sous « games ». Ces jeux inconnus ne doivent pas revenir.
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "presence_sources.json"
            cache.write_text(json.dumps({"games": [10, 20]}), encoding="utf-8")
            sources = PresenceSources(cache, ())
            sources.load_cache()
            self.assertEqual(sources.games, [])
            sources.games = [730, 570]
            sources.save_cache()
            sources = PresenceSources(cache, ())
            sources.load_cache()
        self.assertEqual(sources.games, [730, 570])


def steam_page(*appids: int) -> dict:
    """Page de la recherche Steam, telle que la boutique la renvoie : l'identifiant est dans l'adresse de l'image."""
    return {"desc": "", "items": [
        {"name": f"Jeu {appid}", "logo": f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/"
                                         f"{appid}/abc123/capsule_sm_120.jpg?t=1794333658"} for appid in appids]}


class SteamTopGamesTest(unittest.IsolatedAsyncioTestCase):
    """Les jeux du statut viennent des meilleures ventes de Steam en France, pas de toute la boutique."""

    async def fetch(self, pages: list[dict]) -> tuple[list[int], list[dict]]:
        calls = []

        async def fake_get_json(session, method, url, **kwargs):
            calls.append(kwargs["params"])
            return pages[len(calls) - 1]
        with mock.patch("bot.presence_sources.get_json", fake_get_json):
            return await PresenceSources.fetch_games(None), calls

    async def test_top_500_french_best_sellers(self):
        pages = [steam_page(*range(start + 1, start + 101)) for start in range(0, 700, 100)]
        games, calls = await self.fetch(pages)
        self.assertEqual(games, list(range(1, 501)))
        self.assertEqual(len(calls), 5)  # rien de demandé au-delà du 500e
        self.assertEqual({(c["filter"], c["cc"], c["category1"]) for c in calls}, {("topsellers", "fr", "998")})
        self.assertEqual([c["start"] for c in calls], ["0", "100", "200", "300", "400"])

    async def test_game_moved_between_two_pages_counted_once(self):
        games, _ = await self.fetch([steam_page(*range(1, 101)), steam_page(100, 200)])
        self.assertEqual(games, [*range(1, 101), 200])  # page incomplète : fin du classement

    async def test_unexpected_answer_is_an_error(self):
        # Erreur : la liste précédente est gardée (voir refresh).
        with self.assertRaises(SourceError):
            await self.fetch([{"success": 2}])


class MusicTest(unittest.IsolatedAsyncioTestCase):
    """Une playlist Deezer illisible n'empêche pas de lire les autres."""

    async def run_fetch(self, answers: dict) -> list[Track]:
        urls = {DEEZER_TRACKS_URL.format(p): a for p, a in answers.items()}

        async def fake_get_json(session, method, url, **kwargs):
            answer = urls[url]
            if isinstance(answer, Exception):
                raise answer
            return answer
        sources = PresenceSources(Path("inutilise.json"), deezer_playlists=tuple(answers))
        with mock.patch("bot.presence_sources.get_json", fake_get_json):
            return await sources.fetch_music(None)

    async def test_missing_playlist_is_skipped(self):
        track = {"id": 7, "title": "Titre", "artist": {"name": "Artiste"}, "duration": 200}
        answers = {1: {"data": [track]}, 2: {"error": {"message": "no data"}}, 3: SourceError("réponse HTTP 500")}
        with self.assertLogs("chatbot.presence", "WARNING") as logs:
            tracks = await self.run_fetch(answers)
        self.assertEqual(tracks, [Track("Titre", "Artiste", 200)])
        self.assertTrue(any("playlist 2" in line for line in logs.output))  # l'administrateur sait laquelle corriger
        self.assertTrue(any("playlist 3" in line for line in logs.output))

    async def test_all_playlists_failing_is_an_error(self):
        # Erreur : la liste précédente est gardée (voir refresh).
        with self.assertRaises(SourceError):
            await self.run_fetch({1: {"error": {"message": "no data"}}, 2: SourceError("TimeoutError")})


class RefreshTest(unittest.IsolatedAsyncioTestCase):
    """Une source qui répond de façon inattendue ne fait pas perdre les autres, ni planter le démarrage du statut."""

    async def test_unexpected_answer_keeps_other_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = PresenceSources(Path(tmp) / "presence_sources.json", deezer_playlists=(1,))
            sources.music = [Track("Ancien titre", "Artiste", 180)]

            async def fake_get_json(session, method, url, **kwargs):
                return {"data": [{"id": 7, "title": "Titre", "artist": None}]}  # artiste vide : forme inattendue
            anime = [Anime("Frieren", 28, 24)]
            with mock.patch("bot.presence_sources.get_json", fake_get_json), \
                    mock.patch.object(PresenceSources, "fetch_anime", mock.AsyncMock(return_value=anime)), \
                    mock.patch.object(PresenceSources, "fetch_games", mock.AsyncMock(return_value=[10, 20])), \
                    self.assertLogs("chatbot.presence", "ERROR"):
                await sources.refresh()
            self.assertEqual(sources.anime, anime)
            self.assertEqual(sources.music, [Track("Ancien titre", "Artiste", 180)])  # liste précédente gardée
            self.assertEqual(sources.games, [10, 20])
            self.assertTrue(sources.cache_file.exists())


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
        set_status.assert_awaited_once_with(Status(Kind.ANIME, title="Frieren", episode=5), False)

    async def test_loop_draws_after_a_finished_status(self):
        self.save(self.episodes(3)[0][0], 10)
        manager = self.restore(T0 + timedelta(hours=1))
        draw, set_status = await self.step(manager, T0 + timedelta(hours=1))
        draw.assert_awaited_once()
        set_status.assert_awaited_once_with(Status(Kind.ONLINE), False)  # fin normale, pas un /skip

    async def test_skip_marks_the_status_as_abandoned(self):
        # /skip pendant l'épisode 4 d'un programme qui va jusqu'au 5 : l'anime n'est pas fini, il est abandonné.
        self.save(self.episodes(3)[0][0], 10, self.episodes(4, 5))
        manager = self.restore(T0 + timedelta(minutes=20))
        draw = mock.AsyncMock(return_value=[(Status(Kind.ONLINE), 600)])
        manager.bot = mock.Mock()
        with mock.patch.object(manager, "draw", draw), mock.patch.object(manager, "apply", mock.AsyncMock()),                 mock.patch.object(manager, "save"), frozen_at(T0 + timedelta(minutes=20)):
            await manager.skip()
        manager.bot.dispatch.assert_called_once_with(
            "status_change", Status(Kind.ANIME, title="Frieren", episode=4), Status(Kind.ONLINE), True)


if __name__ == "__main__":
    unittest.main()
