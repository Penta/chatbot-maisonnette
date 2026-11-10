"""Tests des fichiers texte joints : plusieurs par message, 100 Ko au total.
Lancer depuis la racine du projet : python -m unittest"""

import unittest
from types import SimpleNamespace
from unittest import mock

from bot.attachments import AttachmentReader
from bot.discord_bot import ChatBot
from bot.text_files import TextFileError


def text_file(name: str, content: str, size: int | None = None) -> SimpleNamespace:
    data = content.encode("utf-8")
    return SimpleNamespace(filename=name, content_type="text/plain", size=len(data) if size is None else size,
                           read=mock.AsyncMock(return_value=data))


class SeveralTextFilesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reader = AttachmentReader(SimpleNamespace(max_image_size=10_000_000, pdf_max_pages=10))

    async def test_whole_site_read(self):
        # Penta renvoie index.html, style.css et script.js d'un coup : les trois sont lus, dans l'ordre.
        files = [text_file("index.html", "<p>page</p>"), text_file("style.css", "p{}"), text_file("script.js", "go()")]
        images, texts = await self.reader.read(files)
        self.assertEqual(images, [])
        self.assertEqual(texts, [("index.html", "<p>page</p>"), ("style.css", "p{}"), ("script.js", "go()")])

    async def test_total_over_the_limit_refused_before_download(self):
        # Deux fichiers de 60 Ko : chacun passe seul, mais pas ensemble. Rien n'est téléchargé.
        files = [text_file("a.txt", "a", size=60_000), text_file("b.txt", "b", size=60_000)]
        with self.assertRaises(TextFileError) as caught:
            await self.reader.read(files)
        self.assertIn("100 Ko au total", caught.exception.user_message)
        for file in files:
            file.read.assert_not_awaited()

    async def test_single_file_over_the_limit(self):
        with self.assertRaises(TextFileError) as caught:
            await self.reader.read([text_file("gros.txt", "x", size=100_001)])
        self.assertEqual(caught.exception.user_message, "Ce fichier texte est trop long (maximum 100 Ko).")

    async def test_no_attachment(self):
        self.assertEqual(await self.reader.read([]), ([], []))


class MainChannelTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_file_given_to_claude_and_named_in_memory(self):
        bot = mock.Mock()
        bot.read_forward = mock.AsyncMock(return_value=(None, []))
        bot.read_attachments = mock.AsyncMock(return_value=([], [("index.html", "<p>page</p>"), ("style.css", "p{}")]))
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.read_reply = mock.AsyncMock(return_value=(None, []))
        bot.timestamp.return_value = "10/11/2026 18:00"
        bot.message_body.return_value = "corrige le menu"
        message = mock.Mock(attachments=[mock.Mock(), mock.Mock()], stickers=[])
        message.author.display_name = message.author.name = "Penta"
        pending = await ChatBot.read_main_message(bot, message, "corrige le menu")
        self.assertIn("<p>page</p>", pending.extra)
        self.assertIn("p{}", pending.extra)
        self.assertEqual(len(pending.line["attachments"]), 2)
        self.assertIn("index.html", pending.line["attachments"][0])
        self.assertIn("style.css", pending.line["attachments"][1])


class MentionTest(unittest.IsolatedAsyncioTestCase):
    async def test_every_text_file_of_a_mention_read(self):
        bot = mock.Mock()
        bot.settings.pdf_max_pages = 10
        bot.attachments = AttachmentReader(SimpleNamespace(max_image_size=10_000_000, pdf_max_pages=10))
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.read_reply = mock.AsyncMock(return_value=(None, []))
        bot.timestamp.return_value = "10/11/2026 18:00"
        message = mock.Mock(attachments=[text_file("index.html", "<p>page</p>"), text_file("style.css", "p{}")],
                            clean_content="@Rika regarde", guild=None)
        message.author.display_name = message.author.name = "Penta"
        bot.user.name = "Rika"
        prompt, images = await ChatBot.mention_part(bot, message)
        self.assertIn("<p>page</p>", prompt)
        self.assertIn("p{}", prompt)


if __name__ == "__main__":
    unittest.main()
