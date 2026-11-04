"""Tests des textes envoyés à Claude et aux membres. Lancer depuis la racine du projet : python -m unittest"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bot.claude_client import ClaudeClient
from bot.prompts import DEFAULT_PERSONALITY, MEMBER_MEMORY_NOTE


class PromptsTest(unittest.TestCase):
    def test_memory_note_formats(self):
        # Une accolade de trop ou un champ oublié ferait planter chaque message avec MEMORY=on.
        note = MEMBER_MEMORY_NOTE.format(member_limit=3000, server_limit=5000, max_nicknames=5, max_nickname_length=32,
                                         max_lines=25)
        self.assertNotIn("{", note)
        self.assertIn("ne récite jamais la fiche", note)


class PersonalityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "personality_prompt.txt"
        # Sans __init__ : pas de client Anthropic à créer.
        self.client = ClaudeClient.__new__(ClaudeClient)
        self.client.settings = SimpleNamespace(personality_file=self.path)
        self.client._personality_cache = None

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, data: bytes, mtime: int):
        self.path.write_bytes(data)
        os.utime(self.path, (mtime, mtime))  # deux écritures rapprochées auraient la même date

    def test_not_utf8_keeps_last_valid(self):
        self.write("Tu es Rika, très gentille.".encode("utf-8"), 1000)
        self.assertEqual(self.client.personality(), "Tu es Rika, très gentille.")
        self.write("Tu es Rika, très méchante.".encode("cp1252"), 2000)
        with self.assertLogs("chatbot.claude", "ERROR") as logs:
            self.assertEqual(self.client.personality(), "Tu es Rika, très gentille.")
        self.assertIn("UTF-8", logs.output[0])
        self.assertIn("dernière valide", logs.output[0])
        with self.assertNoLogs("chatbot.claude", "ERROR"):  # une seule erreur par version du fichier
            self.client.personality()
        self.write("Tu es Rika, réparée.".encode("utf-8"), 3000)
        self.assertEqual(self.client.personality(), "Tu es Rika, réparée.")

    def test_not_utf8_at_start_uses_default(self):
        self.write("Tu es Rika, très méchante.".encode("cp1252"), 1000)
        with self.assertLogs("chatbot.claude", "ERROR") as logs:
            self.assertEqual(self.client.personality(), DEFAULT_PERSONALITY)
        self.assertIn("par défaut", logs.output[0])

    def test_directory(self):
        self.path.mkdir()
        with self.assertLogs("chatbot.claude", "ERROR") as logs:
            self.assertEqual(self.client.personality(), DEFAULT_PERSONALITY)
        self.assertIn("est un dossier", logs.output[0])


if __name__ == "__main__":
    unittest.main()
