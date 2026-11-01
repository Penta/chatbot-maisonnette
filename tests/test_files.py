"""Tests de l'écriture des fichiers. Lancer depuis la racine du projet : python -m unittest"""

import tempfile
import unittest
from pathlib import Path

from bot.files import write_atomic
from bot.memory import MemoryStore


class WriteAtomicTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_and_creates_directory(self):
        path = self.root / "a" / "b.md"
        write_atomic(path, "habite à Lyon\n")
        self.assertEqual(path.read_text(encoding="utf-8"), "habite à Lyon\n")
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_temporary_file_removed_on_failure(self):
        path = self.root / "b.md"
        path.mkdir()  # un dossier à ce nom : le renommage échoue
        with self.assertRaises(OSError):
            write_atomic(path, "habite à Lyon")
        self.assertEqual(list(self.root.iterdir()), [path])


class NormalizeTest(unittest.TestCase):
    def test_accepted_spellings(self):
        for name in ("membres/123.md", "membres/123", "/memories/membres/123.md", "memories/membres/123.md",
                     "memories\\membres\\123.md"):
            self.assertEqual(MemoryStore._normalize(name), "membres/123.md", name)
        self.assertEqual(MemoryStore._normalize("memories/serveur.md"), "serveur.md")

    def test_no_way_out(self):
        # Le reste est refusé par _check : seuls serveur.md et membres/<id>.md existent.
        self.assertEqual(MemoryStore._normalize("memories/../registre.json"), "../registre.json")


if __name__ == "__main__":
    unittest.main()
