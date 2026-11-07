"""Tests du registre des membres. Lancer depuis la racine du projet : python -m unittest"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.members import FILE_VERSION, MemberRegistry

ENTRY = {"username": "bob", "global_name": "Bob", "nickname": None, "former_names": [],
         "memory": False, "first_seen": "2026-10-31"}


class BrokenRegistryTest(unittest.IsolatedAsyncioTestCase):
    """Registre illisible : rien n'est oublié, la mémoire est coupée jusqu'à sa réparation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "registre.json"

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, members: dict | None = None, text: str | None = None) -> MemberRegistry:
        if text is None:
            text = json.dumps({"version": FILE_VERSION, "members": members})
        self.path.write_text(text, encoding="utf-8")
        return MemberRegistry(self.path)

    async def assert_broken_and_kept(self, registry: MemberRegistry) -> None:
        before = self.path.read_bytes()
        self.assertIsNotNone(registry.broken)
        await registry.see(222, "alice", "Alice")  # rien n'est écrit
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])  # ni copie, ni fichier temporaire

    async def test_truncated_file(self):
        text = json.dumps({"version": FILE_VERSION, "members": {"111": ENTRY}})
        with self.assertLogs("chatbot.members", "ERROR"):
            registry = self.load(text=text[:-20])
        await self.assert_broken_and_kept(registry)

    async def test_unknown_version(self):
        # Les registres d'avant la 5.11 (version 1, ou sans version) ne sont plus convertis.
        for version in (1, "3", None, 4):
            with self.subTest(version=version):
                text = json.dumps({"version": version, "members": {"111": ENTRY}})
                with self.assertLogs("chatbot.members", "ERROR"):
                    registry = self.load(text=text)
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
        self.path.write_text(json.dumps({"version": FILE_VERSION, "members": {"111": ENTRY}}), encoding="utf-8-sig")
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
        with mock.patch("bot.members.write_atomic", side_effect=OSError("disque plein")),                 self.assertLogs("chatbot.members", "ERROR"):
            self.assertFalse(await self.registry.set_memory(111, True))
        self.assertTrue(self.registry.memory_enabled(111))  # appliqué quand même
        self.assertFalse(self.saved_memory())
        # Même réglage redemandé : la sauvegarde est retentée.
        self.assertTrue(await self.registry.set_memory(111, True))
        self.assertTrue(self.saved_memory())

    async def test_any_later_save_catches_up(self):
        with mock.patch("bot.members.write_atomic", side_effect=OSError("disque plein")),                 self.assertLogs("chatbot.members", "ERROR"):
            await self.registry.set_memory(111, True)
        await self.registry.see(222, "alice", "Alice")
        self.assertTrue(self.saved_memory())

    async def test_unchanged_setting_not_rewritten(self):
        with mock.patch("bot.members.write_atomic") as write:
            self.assertTrue(await self.registry.set_memory(111, False))
        write.assert_not_called()

    async def test_disabled_by_default(self):
        self.assertFalse(self.registry.memory_enabled(111))
        self.assertFalse(self.registry.memory_enabled(999))  # membre inconnu
        self.assertFalse(self.saved_memory())


class OldFilesRemovalTest(unittest.TestCase):
    """Registre en version 2 : les fiches des membres qui n'ont pas activé la mémoire sont effacées, une fois."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "registre.json"
        self.files = self.path.parent / "membres"
        self.files.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, version: int, members: dict, files: list[str]) -> MemberRegistry:
        for name in files:
            (self.files / name).write_text("habite à Lyon", encoding="utf-8")
        self.path.write_text(json.dumps({"version": version, "members": members}), encoding="utf-8")
        return MemberRegistry(self.path)

    def remaining(self) -> list[str]:
        return sorted(path.name for path in self.files.iterdir())

    def test_files_removed_once(self):
        members = {"1": {**ENTRY, "memory": True}, "2": {**ENTRY, "memory": False}}
        with self.assertLogs("chatbot.members", "INFO") as logs:
            registry = self.load(2, members, ["1.md", "2.md", "3.md", "notes.md"])
        # Gardées : la fiche d'un membre qui a activé la mémoire, et un fichier qui n'est pas une fiche.
        self.assertEqual(self.remaining(), ["1.md", "notes.md"])
        self.assertTrue(any("2 ancienne(s) fiche(s) effacée(s)" in line for line in logs.output))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], FILE_VERSION)
        self.assertTrue(registry.memory_enabled("1"))
        # Converti : une fiche apparue depuis n'est plus effacée au démarrage suivant.
        (self.files / "2.md").write_text("x", encoding="utf-8")
        MemberRegistry(self.path)
        self.assertEqual(self.remaining(), ["1.md", "2.md", "notes.md"])

    def test_current_version_untouched(self):
        self.load(FILE_VERSION, {"2": {**ENTRY, "memory": False}}, ["2.md"])
        self.assertEqual(self.remaining(), ["2.md"])

    def test_broken_registry_removes_nothing(self):
        # Registre illisible : personne n'aurait la mémoire activée, toutes les fiches seraient effacées.
        for name in ("1.md", "2.md"):
            (self.files / name).write_text("habite à Lyon", encoding="utf-8")
        self.path.write_text('{"version": 2, "members": ', encoding="utf-8")
        with self.assertLogs("chatbot.members", "ERROR"):
            registry = MemberRegistry(self.path)
        self.assertIsNotNone(registry.broken)
        self.assertEqual(self.remaining(), ["1.md", "2.md"])

    def test_removal_failure_retried(self):
        with mock.patch("pathlib.Path.unlink", side_effect=OSError("droits")),                 self.assertLogs("chatbot.members", "ERROR"):
            self.load(2, {"2": {**ENTRY, "memory": False}}, ["2.md"])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], 2)  # refait au démarrage
        MemberRegistry(self.path)
        self.assertEqual(self.remaining(), [])


if __name__ == "__main__":
    unittest.main()
