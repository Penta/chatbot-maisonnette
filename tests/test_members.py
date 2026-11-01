"""Tests du registre des membres. Lancer depuis la racine du projet : python -m unittest"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class SaveFailureTest(unittest.IsolatedAsyncioTestCase):
    """Registre impossible à écrire (disque plein, droits) : set_memory le signale."""

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "registre.json"
        self.registry = MemberRegistry(self.path)
        await self.registry.see(111, "bob", "Bob")

    async def asyncTearDown(self):
        self._tmp.cleanup()

    def saved_memory(self) -> bool:
        return json.loads(self.path.read_text(encoding="utf-8"))["members"]["111"]["memory"]

    async def test_failure_reported_then_caught_up(self):
        with mock.patch("bot.members.write_atomic", side_effect=OSError("disque plein")), \
                self.assertLogs("chatbot.members", "ERROR"):
            self.assertFalse(await self.registry.set_memory(111, False))
        self.assertFalse(self.registry.memory_enabled(111))  # appliqué quand même
        self.assertTrue(self.saved_memory())
        # Même réglage redemandé : la sauvegarde est retentée.
        self.assertTrue(await self.registry.set_memory(111, False))
        self.assertFalse(self.saved_memory())

    async def test_any_later_save_catches_up(self):
        with mock.patch("bot.members.write_atomic", side_effect=OSError("disque plein")), \
                self.assertLogs("chatbot.members", "ERROR"):
            await self.registry.set_memory(111, False)
        await self.registry.see(222, "alice", "Alice")
        self.assertFalse(self.saved_memory())

    async def test_unchanged_setting_not_rewritten(self):
        with mock.patch("bot.members.write_atomic") as write:
            self.assertTrue(await self.registry.set_memory(111, True))
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
