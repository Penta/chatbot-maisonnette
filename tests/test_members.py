"""Tests du registre des membres. Lancer depuis la racine du projet : python -m unittest"""

import json
import tempfile
import unittest
from pathlib import Path

from bot.members import MemberRegistry

ENTRY = {"username": "bob", "global_name": "Bob", "nickname": None, "former_names": [],
         "notified": True, "memory": False, "first_seen": "2026-10-31"}


class BrokenRegistryTest(unittest.IsolatedAsyncioTestCase):
    """Registre illisible : rien n'est oublié, la mémoire est coupée jusqu'à sa réparation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "registre.json"

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, members: dict | None = None, text: str | None = None) -> MemberRegistry:
        if text is None:
            text = json.dumps({"version": 1, "members": members})
        self.path.write_text(text, encoding="utf-8")
        return MemberRegistry(self.path)

    async def assert_broken_and_kept(self, registry: MemberRegistry) -> None:
        before = self.path.read_bytes()
        self.assertIsNotNone(registry.broken)
        await registry.see(222, "alice", "Alice")  # rien n'est écrit
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])  # ni copie, ni fichier temporaire

    async def test_truncated_file(self):
        text = json.dumps({"version": 1, "members": {"111": ENTRY}})
        with self.assertLogs("chatbot.members", "ERROR"):
            registry = self.load(text=text[:-20])
        await self.assert_broken_and_kept(registry)

    async def test_malformed_memory_field(self):
        with self.assertLogs("chatbot.members", "ERROR"):
            registry = self.load({"111": {**ENTRY, "memory": "non"}})
        await self.assert_broken_and_kept(registry)

    async def test_missing_memory_field(self):
        entry = dict(ENTRY)
        del entry["memory"]
        with self.assertLogs("chatbot.members", "ERROR"):
            registry = self.load({"111": entry})
        await self.assert_broken_and_kept(registry)

    async def test_entry_without_username(self):
        with self.assertLogs("chatbot.members", "ERROR"):
            registry = self.load({"111": {**ENTRY, "username": None}})
        await self.assert_broken_and_kept(registry)

    async def test_valid_file(self):
        registry = self.load({"111": ENTRY})
        self.assertIsNone(registry.broken)
        self.assertFalse(registry.memory_enabled(111))
        await registry.see(222, "alice", "Alice")
        self.assertIn("222", json.loads(self.path.read_text(encoding="utf-8"))["members"])

    async def test_file_with_bom(self):
        # Registre corrigé à la main puis enregistré avec BOM (Bloc-notes) : toujours lisible.
        self.path.write_text(json.dumps({"version": 1, "members": {"111": ENTRY}}), encoding="utf-8-sig")
        registry = MemberRegistry(self.path)
        self.assertIsNone(registry.broken)
        self.assertFalse(registry.memory_enabled(111))
        await registry.see(222, "alice", "Alice")
        self.assertIn("222", json.loads(self.path.read_text(encoding="utf-8"))["members"])  # réécrit sans BOM

    async def test_missing_file(self):
        registry = MemberRegistry(self.path)
        self.assertIsNone(registry.broken)
        self.assertEqual(registry.members(), {})


if __name__ == "__main__":
    unittest.main()
