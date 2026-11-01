"""Tests des textes envoyés à Claude et aux membres. Lancer depuis la racine du projet : python -m unittest"""

import unittest

from bot.claude_client import MEMBER_MEMORY_NOTE


class PromptsTest(unittest.TestCase):
    def test_memory_note_formats(self):
        # Une accolade de trop ou un champ oublié ferait planter chaque message avec MEMORY=on.
        note = MEMBER_MEMORY_NOTE.format(member_limit=3000, server_limit=5000, max_nicknames=5, max_nickname_length=32)
        self.assertNotIn("{", note)
        self.assertIn("ne récite jamais la fiche", note)


if __name__ == "__main__":
    unittest.main()
