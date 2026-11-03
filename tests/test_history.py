"""Tests de la mémoire des conversations. Lancer depuis la racine du projet : python -m unittest"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from bot.history import (DELETED_MARKER, DELETED_REPLY, EDITED_MARKER, FILE_VERSION, IMAGE_MARKER, WEB_MARKER,
                         ConversationHistory, images_marker, render_line)

CHANNEL = 123
DATE = "lundi 2 novembre 2026 à 14:05"
# Heure figée pour les copies de secours : leur nom ne dépend pas de l'heure de la machine.
NOW = datetime(2026, 1, 2, 3, 4, 5)
STAMP = "2026-01-02_03-04-05"


def line(message_id: int, content: str, author: str = "Bob", **extra) -> dict:
    """Une ligne de la mémoire pour un message Discord, comme le bot la construit."""
    return {"id": message_id, "date": DATE, "author": author, "content": content, **extra}


def said(content: str, author: str = "Bob", **extra) -> str:
    """Texte rendu d'une ligne simple."""
    return render_line({"date": DATE, "author": author, "content": content, **extra})


class HistoryTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "conversation_history.json"
        patcher = mock.patch("bot.history.datetime")
        self.addCleanup(patcher.stop)
        patcher.start().now.return_value = NOW

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, max_length: int = 10) -> ConversationHistory:
        return ConversationHistory(self.path, max_length)

    def write(self, raw) -> bytes:
        """Écrit un fichier d'historique (objet JSON ou texte brut) et renvoie son contenu exact."""
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        self.path.write_text(text, encoding="utf-8")
        return self.path.read_bytes()

    def other_files(self) -> list[str]:
        """Noms des fichiers du dossier, à part l'historique lui-même."""
        return sorted(p.name for p in self.root.iterdir() if p != self.path)

    async def exchanges(self, history: ConversationHistory, count: int, start: int = 0) -> None:
        for i in range(start, start + count):
            await history.add_exchange(CHANNEL, [line(i, f"question {i}")], f"réponse {i}")


class ExchangeTest(HistoryTestCase):
    async def test_get_format(self):
        history = self.load()
        await history.add_exchange(CHANNEL, [line(1, "Salut"), line(2, "ça va ?", author="Alice")], "Oui !")
        self.assertEqual(history.get(CHANNEL), [
            {"role": "user", "content": f"{DATE}, Bob a écrit : Salut\n{DATE}, Alice a écrit : ça va ?"},
            {"role": "assistant", "content": "Oui !"},
        ])

    async def test_web_search_marked(self):
        history = self.load()
        await history.add_exchange(CHANNEL, [line(1, "Quel temps fait-il ?")], "Il pleut.", web_used=True)
        self.assertEqual(history.get(CHANNEL)[0]["content"], f"{said('Quel temps fait-il ?')} {WEB_MARKER}")

    async def test_unknown_channel_empty(self):
        self.assertEqual(self.load().get(CHANNEL), [])
        self.assertFalse(self.path.exists())

    async def test_channel_id_as_text_or_number(self):
        history = self.load()
        await history.add_exchange(CHANNEL, [line(1, "Salut")], "Bonjour")
        self.assertEqual(history.get(str(CHANNEL)), history.get(CHANNEL))

    async def test_channels_kept_apart(self):
        history = self.load()
        await history.add_exchange(1, [line(1, "dans le salon 1")], "réponse 1")
        await history.add_exchange(2, [line(2, "dans le salon 2")], "réponse 2")
        self.assertEqual(history.get(1)[1]["content"], "réponse 1")
        self.assertEqual(history.get(2)[1]["content"], "réponse 2")

    async def test_reloaded_from_file(self):
        history = self.load()
        await history.add_exchange(CHANNEL, [line(1, "Salut", attachments=[IMAGE_MARKER])], "Bonjour")
        await history.add_exchange(CHANNEL, [line(2, "Et la météo ?", reply_to=1, reply="en réponse à Bob : Salut")],
                                   "Il pleut.", web_used=True)
        await history.add_exchange(42, [{"id": 3, "date": DATE, "author": "Alice", "forward": "de #général : coucou"}],
                                   "Merci")
        await history.edit(CHANNEL, 1, "Salut à tous")
        await history.delete(CHANNEL, {2})

        reloaded = self.load()
        self.assertEqual(reloaded.get(CHANNEL), history.get(CHANNEL))
        self.assertEqual(reloaded.get(42), history.get(42))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], FILE_VERSION)
        self.assertEqual(self.other_files(), [])  # ni copie, ni fichier temporaire


class TrimTest(HistoryTestCase):
    """MAX_HISTORY_LENGTH : au moins N messages gardés, jusqu'à la moitié de plus avant d'y revenir d'un coup."""

    async def test_grows_then_cut_back(self):
        history = self.load(max_length=10)  # marge de 4 messages (la moitié, en échanges entiers)
        await self.exchanges(history, 7)
        messages = history.get(CHANNEL)
        self.assertEqual(len(messages), 14)
        self.assertEqual(messages[0]["content"], said("question 0"))  # le début n'a pas bougé

        await self.exchanges(history, 1, start=7)
        messages = history.get(CHANNEL)
        self.assertEqual(len(messages), 10)
        self.assertEqual(messages[0], {"role": "user", "content": said("question 3")})
        self.assertEqual(messages[-1], {"role": "assistant", "content": "réponse 7"})

    async def test_margin_for_each_length(self):
        # La marge est la moitié de N, arrondie à un nombre pair de messages, et au moins un échange.
        for max_length, margin in ((2, 2), (4, 2), (6, 2), (8, 4), (10, 4), (20, 10)):
            with self.subTest(max_length=max_length):
                self.path.unlink(missing_ok=True)
                history = self.load(max_length)
                sizes = []
                for i in range(3 * (max_length + margin)):
                    await self.exchanges(history, 1, start=i)
                    sizes.append(len(history.get(CHANNEL)))
                self.assertEqual(max(sizes), max_length + margin)
                first_cut = sizes.index(max_length + margin) + 1
                self.assertEqual(sizes[first_cut], max_length)
                self.assertGreaterEqual(min(sizes[first_cut:]), max_length)

    async def test_trimmed_on_load(self):
        history = self.load(max_length=20)
        await self.exchanges(history, 10)
        messages = self.load(max_length=10).get(CHANNEL)
        self.assertEqual(messages, history.get(CHANNEL)[-10:])

    async def test_starts_with_user_message(self):
        # Un fichier dont le salon commence par une réponse du bot : elle est retirée au chargement.
        self.write({"version": FILE_VERSION, "channels": {str(CHANNEL): [
            {"role": "assistant", "content": "réponse orpheline"},
            {"role": "user", "lines": [line(1, "Salut")]},
            {"role": "assistant", "content": "Bonjour"},
        ]}})
        self.assertEqual(self.load().get(CHANNEL), [
            {"role": "user", "content": said("Salut")}, {"role": "assistant", "content": "Bonjour"}])


class EditTest(HistoryTestCase):
    async def asyncSetUp(self):
        self.history = self.load()
        await self.history.add_exchange(CHANNEL, [line(1, "Salut"), line(2, "Ça va ?")], "Oui !")

    async def test_edited_message(self):
        self.assertTrue(await self.history.edit(CHANNEL, 1, "Salut à tous"))
        expected = f"{said('Salut à tous')} {EDITED_MARKER}\n{said('Ça va ?')}"
        self.assertEqual(self.history.get(CHANNEL)[0]["content"], expected)
        self.assertEqual(self.load().get(CHANNEL)[0]["content"], expected)  # sauvegardé

    async def test_same_text_not_marked(self):
        # Le texte n'a pas changé (une pièce jointe retirée, par exemple) : rien n'est marqué.
        self.assertFalse(await self.history.edit(CHANNEL, 1, "Salut"))
        self.assertNotIn(EDITED_MARKER, self.history.get(CHANNEL)[0]["content"])

    async def test_unknown_message(self):
        before = self.path.read_bytes()
        self.assertFalse(await self.history.edit(CHANNEL, 999, "nouveau"))
        self.assertEqual(self.path.read_bytes(), before)

    async def test_unknown_channel(self):
        self.assertFalse(await self.history.edit(456, 1, "nouveau"))
        self.assertEqual(self.history.get(456), [])

    async def test_deleted_message_not_edited(self):
        await self.history.delete(CHANNEL, {1})
        self.assertFalse(await self.history.edit(CHANNEL, 1, "revenu"))
        self.assertNotIn("revenu", self.history.get(CHANNEL)[0]["content"])

    async def test_forward_not_edited(self):
        await self.history.add_exchange(CHANNEL, [{"id": 3, "date": DATE, "author": "Bob", "forward": "de #a : b"}],
                                        "Vu")
        self.assertFalse(await self.history.edit(CHANNEL, 3, "texte"))


class DeleteTest(HistoryTestCase):
    async def asyncSetUp(self):
        self.history = self.load()
        await self.history.add_exchange(CHANNEL, [line(1, "Mon secret", attachments=[IMAGE_MARKER])], "Noté !")
        await self.history.add_exchange(
            CHANNEL, [line(2, "Tu as vu ?", author="Alice", reply_to=1, reply="en réponse à Bob : Mon secret")],
            "Oui")

    async def test_deleted_message(self):
        self.assertTrue(await self.history.delete(CHANNEL, {1}))
        messages = self.history.get(CHANNEL)
        self.assertEqual(messages[0]["content"], f"{DATE}, Bob a écrit : {DELETED_MARKER}")
        self.assertEqual(messages[1]["content"], "Noté !")  # la réponse du bot reste

    async def test_reply_to_deleted_message(self):
        await self.history.delete(CHANNEL, {1})
        self.assertEqual(self.history.get(CHANNEL)[2]["content"],
                         f"{DATE}, Alice a écrit ({DELETED_REPLY}) : Tu as vu ?")

    async def test_nothing_left_in_file(self):
        await self.history.delete(CHANNEL, {1})
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("secret", text)
        self.assertNotIn(IMAGE_MARKER, text)
        self.assertEqual(self.load().get(CHANNEL), self.history.get(CHANNEL))

    async def test_several_at_once(self):
        self.assertTrue(await self.history.delete(CHANNEL, {1, 2}))
        messages = self.history.get(CHANNEL)
        self.assertEqual(messages[0]["content"], f"{DATE}, Bob a écrit : {DELETED_MARKER}")
        self.assertEqual(messages[2]["content"], f"{DATE}, Alice a écrit : {DELETED_MARKER}")

    async def test_already_deleted(self):
        await self.history.delete(CHANNEL, {1})
        before = self.path.read_bytes()
        self.assertFalse(await self.history.delete(CHANNEL, {1}))
        self.assertEqual(self.path.read_bytes(), before)

    async def test_unknown_message(self):
        self.assertFalse(await self.history.delete(CHANNEL, {999}))

    async def test_unknown_channel(self):
        self.assertFalse(await self.history.delete(456, {1}))
        self.assertEqual(self.history.get(456), [])


class ResetTest(HistoryTestCase):
    async def test_channel_emptied(self):
        history = self.load()
        await history.add_exchange(CHANNEL, [line(1, "Salut")], "Bonjour")
        await history.add_exchange(456, [line(2, "Ailleurs")], "Ici")
        await history.reset(CHANNEL)
        self.assertEqual(history.get(CHANNEL), [])
        self.assertEqual(len(history.get(456)), 2)  # les autres salons ne bougent pas
        reloaded = self.load()
        self.assertEqual(reloaded.get(CHANNEL), [])
        self.assertEqual(len(reloaded.get(456)), 2)


class BrokenFileTest(HistoryTestCase):
    """Fichier illisible : mis de côté sous un nom daté, et le bot repart d'une mémoire vide."""

    backup_name = f"conversation_history.json.corrompu-{STAMP}"

    async def assert_set_aside(self, content: bytes) -> None:
        with self.assertLogs("chatbot.history", "ERROR"):
            history = self.load()
        self.assertEqual(history.get(CHANNEL), [])
        self.assertFalse(self.path.exists())
        backup = self.root / self.backup_name
        self.assertEqual(backup.read_bytes(), content)
        # La sauvegarde suivante écrit un nouveau fichier, sans toucher à la copie.
        await history.add_exchange(CHANNEL, [line(1, "Salut")], "Bonjour")
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], FILE_VERSION)
        self.assertEqual(backup.read_bytes(), content)

    async def test_invalid_json(self):
        text = json.dumps({"version": FILE_VERSION, "channels": {str(CHANNEL): []}})
        await self.assert_set_aside(self.write(text[:-5]))

    async def test_wrong_encoding(self):
        self.path.write_bytes('{"version": 3, "channels": {"123": [{"role": "user", "content": "été"}]}}'
                              .encode("latin-1"))
        await self.assert_set_aside(self.path.read_bytes())

    async def test_not_a_dictionary(self):
        await self.assert_set_aside(self.write([1, 2, 3]))

    async def test_channels_not_a_dictionary(self):
        await self.assert_set_aside(self.write({"version": FILE_VERSION, "channels": ["123"]}))

    async def test_older_copy_kept(self):
        older = self.root / "conversation_history.json.corrompu-2025-12-31_23-59-59"
        older.write_text("ancienne copie", encoding="utf-8")
        await self.assert_set_aside(self.write("pas du JSON"))
        self.assertEqual(older.read_text(encoding="utf-8"), "ancienne copie")
        self.assertEqual(self.other_files(), [older.name, self.backup_name])


class OldFormatTest(HistoryTestCase):
    """Fichiers des versions précédentes : convertis, et une copie de l'ancien est gardée."""

    def assert_old_copy(self, version: int, content: bytes) -> None:
        self.assertEqual(self.other_files(), [f"conversation_history.json.v{version}-{STAMP}"])
        self.assertEqual((self.root / self.other_files()[0]).read_bytes(), content)
        self.assertEqual(self.path.read_bytes(), content)  # rien n'est réécrit avant la prochaine sauvegarde

    async def test_version_2(self):
        content = self.write({"version": 2, "channels": {str(CHANNEL): [
            {"role": "user", "content": f"{DATE}, Bob a écrit : Regarde [image jointe]"},
            {"role": "assistant", "content": "<think>C'est un chat.</think>\nJoli chat !"},
        ]}})
        with self.assertLogs("chatbot.history", "INFO"):
            history = self.load()
        self.assertEqual(history.get(CHANNEL), [
            {"role": "user", "content": f"{DATE}, Bob a écrit : Regarde {IMAGE_MARKER}"},
            {"role": "assistant", "content": "Joli chat !"},
        ])
        self.assert_old_copy(2, content)

    async def test_mistral_format(self):
        # Premier format, sans numéro de version : un dictionnaire par salon, contenu en blocs.
        content = self.write({str(CHANNEL): {"messages": [
            {"role": "system", "content": "Tu es Rika."},
            {"role": "user", "content": [
                {"type": "text", "text": "Bob : regarde"},
                {"type": "image_url", "image_url": {"url": "https://cdn.discordapp.com/chat.png"}},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Joli chat !"}]},
        ]}})
        with self.assertLogs("chatbot.history", "INFO"):
            history = self.load()
        self.assertEqual(history.get(CHANNEL), [
            {"role": "user", "content": f"Bob : regarde {IMAGE_MARKER}"},
            {"role": "assistant", "content": "Joli chat !"},
        ])
        self.assert_old_copy(1, content)

    async def test_rendered_text_unchanged(self):
        # Le texte envoyé à Claude reste le même après la conversion : le cache de prompt est conservé.
        user = f"{DATE}, Bob a écrit : Quel temps fait-il ? {IMAGE_MARKER} {WEB_MARKER}"
        self.write({"version": 2, "channels": {str(CHANNEL): [
            {"role": "user", "content": user}, {"role": "assistant", "content": "Il pleut."}]}})
        with self.assertLogs("chatbot.history", "INFO"):
            history = self.load()
        self.assertEqual(history.get(CHANNEL), [
            {"role": "user", "content": user}, {"role": "assistant", "content": "Il pleut."}])

    async def test_same_role_merged(self):
        self.write({"version": 2, "channels": {str(CHANNEL): [
            {"role": "user", "content": "un"}, {"role": "user", "content": "deux"},
            {"role": "assistant", "content": "trois"}, {"role": "assistant", "content": "quatre"},
        ]}})
        with self.assertLogs("chatbot.history", "INFO"):
            history = self.load()
        self.assertEqual(history.get(CHANNEL), [
            {"role": "user", "content": "un\ndeux"}, {"role": "assistant", "content": "trois\nquatre"}])

    async def test_unreadable_messages_dropped(self):
        self.write({"version": 2, "channels": {str(CHANNEL): [
            "pas un message", {"role": "tool", "content": "résultat"}, {"role": "user", "content": "  "},
            {"role": "user", "content": None}, {"role": "user", "content": [{"type": "audio"}, "bloc"]},
            {"role": "user", "content": "Salut"},
            {"role": "assistant", "content": "<think>rien d'autre</think>"},
            {"role": "assistant", "content": "Bonjour"},
        ], "456": "pas une liste"}})
        with self.assertLogs("chatbot.history", "INFO"):
            history = self.load()
        self.assertEqual(history.get(CHANNEL), [
            {"role": "user", "content": "Salut"}, {"role": "assistant", "content": "Bonjour"}])
        self.assertEqual(history.get(456), [])

    async def test_converted_file_saved_in_current_format(self):
        self.write({"version": 2, "channels": {str(CHANNEL): [
            {"role": "user", "content": "Salut"}, {"role": "assistant", "content": "Bonjour"}]}})
        with self.assertLogs("chatbot.history", "INFO"):
            history = self.load()
        await history.add_exchange(CHANNEL, [line(1, "Encore moi")], "Re")
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], FILE_VERSION)
        with self.assertNoLogs("chatbot.history"):
            reloaded = self.load()  # plus de conversion, ni de nouvelle copie
        self.assertEqual(reloaded.get(CHANNEL), history.get(CHANNEL))
        self.assertEqual(len(self.other_files()), 1)

    async def test_current_format_not_copied(self):
        self.write({"version": FILE_VERSION, "channels": {str(CHANNEL): [
            {"role": "user", "lines": [line(1, "Salut")]}, {"role": "assistant", "content": "Bonjour"}]}})
        with self.assertNoLogs("chatbot.history"):
            history = self.load()
        self.assertEqual(len(history.get(CHANNEL)), 2)
        self.assertEqual(self.other_files(), [])

    async def test_unreadable_lines_dropped(self):
        # Lignes abîmées dans un fichier au format actuel : seules les lignes lisibles restent.
        self.write({"version": FILE_VERSION, "channels": {str(CHANNEL): [
            {"role": "user", "lines": ["texte", {"author": "Bob", "content": "sans date"}, {"content": ""}]},
            {"role": "user", "lines": [line(1, "Salut"), {"id": 2, "date": DATE, "author": "Bob"}]},
            {"role": "user", "lines": [line(3, "Re")], "web": True},
            {"role": "assistant", "content": "Bonjour"},
        ]}})
        self.assertEqual(self.load().get(CHANNEL), [
            {"role": "user", "content": f"{said('Salut')}\n{said('Re')} {WEB_MARKER}"},
            {"role": "assistant", "content": "Bonjour"},
        ])


class RenderLineTest(unittest.TestCase):
    def test_bot_note(self):
        self.assertEqual(render_line({"content": "[le bot a changé de statut]"}), "[le bot a changé de statut]")

    def test_message(self):
        self.assertEqual(render_line(line(1, "Salut")), f"{DATE}, Bob a écrit : Salut")

    def test_reply(self):
        self.assertEqual(render_line(line(1, "Oui", reply_to=9, reply="en réponse à Alice : Ça va ?")),
                         f"{DATE}, Bob a écrit (en réponse à Alice : Ça va ?) : Oui")
        self.assertEqual(render_line(line(1, "Oui", reply="")), f"{DATE}, Bob a écrit : Oui")

    def test_forward(self):
        forward = {"id": 1, "date": DATE, "author": "Bob", "forward": "de Alice dans #général : coucou"}
        self.assertEqual(render_line(forward), f"{DATE}, Bob a transféré un message de Alice dans #général : coucou")

    def test_edited(self):
        self.assertEqual(render_line(line(1, "Salut", edited=True)), f"{DATE}, Bob a écrit : Salut {EDITED_MARKER}")

    def test_deleted(self):
        deleted = {"id": 1, "date": DATE, "author": "Bob", "deleted": True}
        self.assertEqual(render_line(deleted), f"{DATE}, Bob a écrit : {DELETED_MARKER}")

    def test_attachments(self):
        rendered = render_line(line(1, "Regarde", edited=True, attachments=[IMAGE_MARKER, "[fichier notes.txt]"]))
        self.assertEqual(rendered, f"{DATE}, Bob a écrit : Regarde {EDITED_MARKER} {IMAGE_MARKER} [fichier notes.txt]")

    def test_forward_with_attachments(self):
        forward = {"id": 1, "date": DATE, "author": "Bob", "forward": "de #a : vu", "attachments": [IMAGE_MARKER]}
        self.assertEqual(render_line(forward), f"{DATE}, Bob a transféré un message de #a : vu {IMAGE_MARKER}")


class ImagesMarkerTest(unittest.TestCase):
    def test_one_image(self):
        self.assertEqual(images_marker(1), IMAGE_MARKER)

    def test_several_images(self):
        for count in (2, 5):
            with self.subTest(count=count):
                marker = images_marker(count)
                self.assertTrue(marker.startswith(f"[{count} images jointes : "), marker)
                self.assertIn("elles ne sont plus conservées", marker)


if __name__ == "__main__":
    unittest.main()
