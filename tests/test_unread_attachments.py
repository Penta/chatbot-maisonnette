"""Tests des pièces jointes que Rika ne lit pas : ignorées et signalées, au lieu de faire refuser le message,
dans son salon comme dans les mentions.
Lancer depuis la racine du projet : python -m unittest"""

import unittest
from types import SimpleNamespace
from unittest import mock

from bot.attachments import AttachmentReader, pick
from bot.discord_bot import ChatBot
from bot.media import ImageData


def attachment(name: str, content_type: str, size: int = 1000) -> SimpleNamespace:
    return SimpleNamespace(filename=name, content_type=content_type, size=size, read=mock.AsyncMock(return_value=b"x"))


def video(name: str = "clip.mp4") -> SimpleNamespace:
    return attachment(name, "video/mp4")


def image(name: str) -> SimpleNamespace:
    return attachment(name, "image/png")


def pdf(name: str) -> SimpleNamespace:
    return attachment(name, "application/pdf")


def reader(pdf_max_pages: int = 10) -> AttachmentReader:
    reader = AttachmentReader(SimpleNamespace(max_image_size=10_000_000, pdf_max_pages=pdf_max_pages))
    reader.download_image = mock.AsyncMock(side_effect=lambda a: ImageData("image/png", a.filename.encode()))
    reader.download_pdf = mock.AsyncMock(side_effect=lambda a: ImageData("application/pdf", b"%PDF", a.filename))
    return reader


class PickTest(unittest.TestCase):
    def test_video_signalled(self):
        picked = pick([video()], read_pdfs=True)
        self.assertEqual((picked.images, picked.pdf, picked.texts), ([], None, []))
        self.assertEqual(picked.unread, ["[vidéo jointe « clip.mp4 » : tu ne peux pas voir les vidéos]"])

    def test_video_without_announced_type(self):
        # Discord n'annonce pas toujours le type : l'extension suffit à reconnaître une vidéo.
        for name in ("clip.mp4", "Souvenir.MOV", "extrait.webm", "film.mkv"):
            with self.subTest(name=name):
                picked = pick([attachment(name, None)], read_pdfs=True)
                self.assertEqual(picked.unread, [f"[vidéo jointe « {name} » : tu ne peux pas voir les vidéos]"])

    def test_ts_video_not_read_as_code(self):
        # Un .ts peut être du TypeScript ou une vidéo MPEG-TS : Discord annonce alors video/mp2t. La vidéo,
        # souvent plus grosse que 100 Ko, ne doit pas être lue comme du texte (et faire refuser le message).
        clip = attachment("extrait.ts", "video/mp2t", size=5_000_000)
        picked = pick([clip], read_pdfs=True)
        self.assertEqual(picked.texts, [])
        self.assertEqual(picked.unread, ["[vidéo jointe « extrait.ts » : tu ne peux pas voir les vidéos]"])

    def test_announced_audio_not_read_as_text(self):
        picked = pick([attachment("chanson.ts", "audio/mp2t")], read_pdfs=True)
        self.assertEqual(picked.texts, [])
        self.assertEqual(len(picked.unread), 1)

    def test_typescript_still_read(self):
        for content_type in ("text/plain", "application/typescript", None):
            with self.subTest(content_type=content_type):
                code = attachment("app.ts", content_type)
                self.assertEqual(pick([code], read_pdfs=True).texts, [code])

    def test_extra_images_counted(self):
        shots = [image(f"capture{i}.png") for i in range(6)]
        picked = pick(shots, read_pdfs=True)
        self.assertEqual(picked.images, shots[:4])
        self.assertEqual(picked.unread, ["[2 autres images jointes, que tu n'as pas vues : 4 au maximum par message]"])

    def test_pdf_next_to_an_image_signalled(self):
        photo, doc = image("photo.png"), pdf("doc.pdf")
        picked = pick([photo, doc], read_pdfs=True)
        self.assertEqual((picked.images, picked.pdf), ([photo], None))
        self.assertEqual(picked.unread, ["[PDF joint « doc.pdf », que tu n'as pas lu : un seul PDF par message, "
                                         "et seulement sans image]"])

    def test_second_pdf_signalled(self):
        first, second = pdf("a.pdf"), pdf("b.pdf")
        picked = pick([first, second], read_pdfs=True)
        self.assertIs(picked.pdf, first)
        self.assertEqual(len(picked.unread), 1)
        self.assertIn("b.pdf", picked.unread[0])

    def test_pdf_when_pdfs_are_off(self):
        picked = pick([pdf("doc.pdf")], read_pdfs=False)
        self.assertIsNone(picked.pdf)
        self.assertEqual(picked.unread, ["[fichier joint « doc.pdf » : tu ne peux pas ouvrir ce type de fichier]"])


class MainChannelReadTest(unittest.IsolatedAsyncioTestCase):
    async def test_video_no_longer_refused(self):
        # « Regarde ce clip ! » + clip.mp4 : avant, le message entier était refusé.
        clip = video()
        images, texts, unread = await reader().read([clip])
        self.assertEqual((images, texts), ([], []))
        self.assertIn("clip.mp4", unread[0])
        clip.read.assert_not_awaited()

    async def test_six_screenshots_read_four(self):
        images, _, unread = await reader().read([image(f"capture{i}.png") for i in range(6)])
        self.assertEqual(len(images), 4)
        self.assertIn("2 autres images", unread[0])

    async def test_image_and_pdf_read_the_image(self):
        r = reader()
        images, _, unread = await r.read([image("photo.png"), pdf("doc.pdf")])
        self.assertEqual([i.data for i in images], [b"photo.png"])
        r.download_pdf.assert_not_awaited()
        self.assertIn("doc.pdf", unread[0])

    async def test_note_given_to_claude_and_kept_in_memory(self):
        bot = mock.Mock()
        bot.read_forward = mock.AsyncMock(return_value=(None, []))
        note = "[vidéo jointe « clip.mp4 » : tu ne peux pas voir les vidéos]"
        bot.read_attachments = mock.AsyncMock(return_value=([], [], [note]))
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.read_reply = mock.AsyncMock(return_value=(None, [], []))
        bot.timestamp.return_value = "11/11/2026 18:00"
        bot.message_body.return_value = "Regarde ce clip !"
        message = mock.Mock(attachments=[video()], stickers=[])
        message.author.display_name = message.author.name = "Lucas"
        pending = await ChatBot.read_main_message(bot, message, "Regarde ce clip !")
        self.assertIn(note, pending.user_text)
        self.assertEqual(pending.line["attachments"], [note])


class MentionReadTest(unittest.IsolatedAsyncioTestCase):
    async def test_video_signalled_in_a_mention(self):
        bot = mock.Mock()
        bot.settings.pdf_max_pages = 10
        bot.attachments = reader()
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.read_reply = mock.AsyncMock(return_value=(None, [], []))
        bot.timestamp.return_value = "11/11/2026 18:00"
        message = mock.Mock(attachments=[video()], clean_content="@Rika regarde ça", guild=None)
        message.author.display_name = message.author.name = "Lucas"
        bot.user.name = "Rika"
        prompt, images = await ChatBot.mention_part(bot, message)
        self.assertEqual(images, [])
        self.assertIn("[vidéo jointe « clip.mp4 » : tu ne peux pas voir les vidéos]", prompt)



class SameOrderTest(unittest.IsolatedAsyncioTestCase):
    """Penta envoie clip.mp4 et index.html en citant un message qui porte notes.txt : dans le salon comme dans une
    mention, Claude reçoit la note sur la vidéo juste après le texte, puis les fichiers du message, puis ceux du
    message cité."""

    NOTE = "[vidéo jointe « clip.mp4 » : tu ne peux pas voir les vidéos]"

    def assert_order(self, text: str):
        positions = [text.index(self.NOTE), text.index("<p>page</p>"), text.index("à faire")]
        self.assertEqual(positions, sorted(positions))

    async def test_main_channel(self):
        bot = mock.Mock()
        bot.read_forward = mock.AsyncMock(return_value=(None, []))
        bot.read_attachments = mock.AsyncMock(return_value=([], [("index.html", "<p>page</p>")], [self.NOTE]))
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.read_reply = mock.AsyncMock(return_value=("en réponse", [], [("notes.txt", "à faire")]))
        bot.timestamp.return_value = "11/11/2026 18:00"
        bot.message_body.return_value = "regarde"
        message = mock.Mock(attachments=[], stickers=[])
        message.author.display_name = message.author.name = "Penta"
        pending = await ChatBot.read_main_message(bot, message, "regarde")
        self.assert_order(pending.user_text)

    async def test_mention(self):
        bot = mock.Mock()
        bot.settings.pdf_max_pages = 10
        bot.attachments = reader()
        bot.attachments.download_text_files = mock.AsyncMock(return_value=[("index.html", "<p>page</p>")])
        bot.read_gif = mock.AsyncMock(return_value=None)
        bot.read_reply = mock.AsyncMock(return_value=("en réponse", [], [("notes.txt", "à faire")]))
        bot.timestamp.return_value = "11/11/2026 18:00"
        page = attachment("index.html", "text/html")
        message = mock.Mock(attachments=[video(), page], clean_content="@Rika regarde", guild=None)
        message.author.display_name = message.author.name = "Penta"
        bot.user.name = "Rika"
        prompt, _ = await ChatBot.mention_part(bot, message)
        self.assert_order(prompt)

if __name__ == "__main__":
    unittest.main()
