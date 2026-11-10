"""Tests de la configuration (MAX_TOKENS, mémoire et fichiers écrits). Lancer depuis la racine du projet : python -m unittest"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.config import ConfigError, load_settings, memory_max_tokens, memory_warning

BASE = {"DISCORD_TOKEN": "x", "ANTHROPIC_API_KEY": "x", "CHANNEL_ID": "1",
        "CLAUDE_MODEL": "claude-sonnet-5-5", "MEMORY": "on", "MEMORY_SERVER_LIMIT": "5000"}


def settings(**env):
    # Sans le .env du projet : seulement les variables données.
    with mock.patch.dict("os.environ", {**BASE, **env}, clear=True), mock.patch("bot.config.load_dotenv"):
        return load_settings()


class MemoryMaxTokensTest(unittest.TestCase):
    def test_margin_follows_effort(self):
        # 5000 caractères : 2000 tokens de contenu, plus 2000 pour le reste de l'appel.
        self.assertEqual(memory_max_tokens(settings(THINKING="off")), (4000, 4000))
        self.assertEqual(memory_max_tokens(settings(THINKING="on", EFFORT="medium")), (4000, 6000))
        self.assertEqual(memory_max_tokens(settings(THINKING="on")), (4000, 8000))
        self.assertEqual(memory_max_tokens(settings(THINKING="on", EFFORT="max")), (4000, 12000))

    def test_opus_thinks_even_with_thinking_off(self):
        s = settings(CLAUDE_MODEL="claude-opus-5-5", THINKING="off", EFFORT="low")
        self.assertEqual(memory_max_tokens(s), (4000, 6000))

    def test_too_small_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            settings(MAX_TOKENS="3999")
        self.assertIn("au moins 4000", str(ctx.exception))

    def test_memory_off_not_checked(self):
        self.assertEqual(settings(MEMORY="off", MAX_TOKENS="1000").max_tokens, 1000)
        self.assertIsNone(memory_warning(settings(MEMORY="off", MAX_TOKENS="1000", THINKING="on")))

    def test_warning_when_reflection_does_not_fit(self):
        # Le scénario de l'audit : pile le minimum, mais la réflexion à effort high n'y tient pas.
        s = settings(MAX_TOKENS="4000", THINKING="on", EFFORT="high")
        self.assertIn("au moins 8000", memory_warning(s))
        self.assertIsNone(memory_warning(settings(MAX_TOKENS="8000", THINKING="on", EFFORT="high")))
        self.assertIsNone(memory_warning(settings(MAX_TOKENS="4000", THINKING="off")))



class OutputPathsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_existing_folders_accepted(self):
        s = settings(LOG_FILE=str(self.root / "bot.log"), HISTORY_FILE=str(self.root / "h.json"))
        self.assertEqual(s.memory_dir, self.root / "memories")

    def test_missing_folder_refused(self):
        # Le scénario de l'audit : volume non monté sur /data.
        for name in ("LOG_FILE", "HISTORY_FILE", "MEMORY_DIR"):
            with self.subTest(name), self.assertRaises(ConfigError) as ctx:
                settings(**{name: str(self.root / "absent" / "x")})
            self.assertIn(f"Le dossier de {name} n'existe pas", str(ctx.exception))

    def test_memory_dir_unchecked_when_off(self):
        settings(MEMORY="off", MEMORY_DIR=str(self.root / "absent" / "x"))

    def test_file_path_is_folder_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            settings(HISTORY_FILE=str(self.root))
        self.assertIn("HISTORY_FILE désigne un dossier", str(ctx.exception))

    def test_unwritable_folder_refused(self):
        with mock.patch("bot.config.tempfile.TemporaryFile", side_effect=PermissionError(13, "Permission denied")),                 self.assertRaises(ConfigError) as ctx:
            settings()
        self.assertIn("Impossible d'écrire dans le dossier de LOG_FILE", str(ctx.exception))


class MaximumTest(unittest.TestCase):
    def test_values_above_maximum_refused(self):
        for name, value in (("MAX_TOKENS", "64001"), ("WEB_MAX_USES", "11"), ("MEMORY_MAX_OPERATIONS", "21"),
                            ("PDF_MAX_PAGES", "101"), ("PRESENCE_SPEED", "100.5")):
            with self.subTest(name), self.assertRaises(ConfigError) as ctx:
                settings(MEMORY="off", **{name: value})
            self.assertIn(f"{name} ne peut pas dépasser", str(ctx.exception))

    def test_maximum_accepted(self):
        s = settings(MEMORY="off", MAX_TOKENS="64000", WEB_MAX_USES="10", MEMORY_MAX_OPERATIONS="20",
                     PDF_MAX_PAGES="100", PRESENCE_SPEED="100")
        self.assertEqual((s.max_tokens, s.pdf_max_pages, s.presence_speed), (64000, 100, 100))


class ReplyPauseTest(unittest.TestCase):
    def test_default(self):
        self.assertEqual(settings(MEMORY="off").reply_pause, 2)

    def test_limits_accepted(self):
        self.assertEqual(settings(MEMORY="off", REPLY_PAUSE="0").reply_pause, 0)
        self.assertEqual(settings(MEMORY="off", REPLY_PAUSE="1.5").reply_pause, 1.5)
        self.assertEqual(settings(MEMORY="off", REPLY_PAUSE="10").reply_pause, 10)

    def test_values_out_of_limits_refused(self):
        for value in ("-1", "10.5", "deux"):
            with self.subTest(value), self.assertRaises(ConfigError) as ctx:
                settings(MEMORY="off", REPLY_PAUSE=value)
            self.assertIn("REPLY_PAUSE", str(ctx.exception))


class AllErrorsTest(unittest.TestCase):
    def test_every_error_reported_at_once(self):
        # Le scénario de l'audit : trois variables fausses, un seul redémarrage pour les voir toutes.
        with self.assertRaises(ConfigError) as ctx:
            settings(CHANNEL_ID="abc", THINKING="oui", PDF_MAX_PAGES="101", DISCORD_TOKEN="")
        self.assertEqual(len(ctx.exception.errors), 4)
        for text in ("CHANNEL_ID doit être un nombre entier", "THINKING doit valoir", "PDF_MAX_PAGES ne peut pas",
                     "DISCORD_TOKEN est obligatoire"):
            self.assertIn(text, str(ctx.exception))

    def test_empty_quotes_count_as_missing(self):
        # docker run --env-file garde les guillemets : DISCORD_TOKEN="" ne doit pas passer pour une valeur.
        for name in ("DISCORD_TOKEN", "ANTHROPIC_API_KEY", "CHANNEL_ID"):
            for raw in ('""', "''", '" "'):
                with self.subTest(name=name, raw=raw), self.assertRaises(ConfigError) as ctx:
                    settings(**{name: raw})
                self.assertIn(f"{name} est obligatoire", str(ctx.exception))

    def test_invalid_value_not_reused_in_other_checks(self):
        # MAX_TOKENS illisible : pas de second message sur la mémoire, ni pour THINKING invalide avec EFFORT=max.
        with self.assertRaises(ConfigError) as ctx:
            settings(MAX_TOKENS="beaucoup", THINKING="oui", EFFORT="max")
        self.assertEqual(len(ctx.exception.errors), 2)

    def test_invalid_effort_with_memory(self):
        # EFFORT inconnu avec la mémoire : un message clair, pas une KeyError dans memory_max_tokens.
        for env in ({"THINKING": "on"}, {"CLAUDE_MODEL": "claude-opus-5-5"}):
            with self.subTest(**env), self.assertRaises(ConfigError) as ctx:
                settings(EFFORT="ultra", **env)
            self.assertEqual(len(ctx.exception.errors), 1)
            self.assertIn("EFFORT doit valoir", str(ctx.exception))

    def test_log_file_given_when_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "bot.log"
            with self.assertRaises(ConfigError) as ctx:
                settings(CHANNEL_ID="abc", LOG_FILE=str(log))
            self.assertEqual(ctx.exception.log_file, log)
            with self.assertRaises(ConfigError) as ctx:
                settings(LOG_FILE=str(Path(folder) / "absent" / "bot.log"))
            self.assertIsNone(ctx.exception.log_file)

    def test_log_file_off_and_level(self):
        s = settings(LOG_FILE="OFF", LOG_LEVEL="debug")
        self.assertIsNone(s.log_file)
        self.assertEqual(s.log_level, "DEBUG")
        with self.assertRaises(ConfigError) as ctx:
            settings(LOG_LEVEL="bavard")
        self.assertIn("LOG_LEVEL", str(ctx.exception))


class PresenceWeightsTest(unittest.TestCase):
    ONLY_GAME = "anime=0,musique=0,pensee=0,jeu=100,en_ligne=0,ne_pas_deranger=0,absent=0"

    def test_empty_items_ignored(self):
        s = settings(MEMORY="off", PRESENCE_WEIGHTS="anime=5,,musique=7,")
        self.assertEqual((s.presence_weights["anime"], s.presence_weights["musique"]), (5, 7))

    def test_duplicate_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            settings(MEMORY="off", PRESENCE_WEIGHTS="anime=5,Anime=0")
        self.assertIn("le statut anime apparaît plusieurs fois", str(ctx.exception))

    def test_only_game_accepted(self):
        settings(MEMORY="off", PRESENCE_WEIGHTS=self.ONLY_GAME)


if __name__ == "__main__":
    unittest.main()
