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


if __name__ == "__main__":
    unittest.main()
