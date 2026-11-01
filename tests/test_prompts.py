"""Tests des textes envoyés à Claude et aux membres. Lancer depuis la racine du projet : python -m unittest"""

import unittest

from bot.claude_client import MEMBER_MEMORY_NOTE
from bot.discord_bot import WELCOME


class PromptsTest(unittest.TestCase):
    def test_memory_note_formats(self):
        # Une accolade de trop ou un champ oublié ferait planter chaque message avec MEMORY=on.
        note = MEMBER_MEMORY_NOTE.format(member_limit=3000, server_limit=5000, max_nicknames=5, max_nickname_length=32)
        self.assertNotIn("{", note)
        self.assertIn("ne récite jamais la fiche", note)

    def test_welcome_formats(self):
        text = WELCOME.format(name="Rika")
        self.assertIn("jamais récitée aux autres", text)
        self.assertIn("administrateurs du serveur peuvent la voir", text)


if __name__ == "__main__":
    unittest.main()
