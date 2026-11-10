"""Tests des GIF choisis dans le sélecteur de Discord (Klipy, Giphy…). Lancer depuis la racine du projet :
python -m unittest"""

import asyncio
import io
import unittest
from datetime import datetime, timezone
from unittest import mock

import discord
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

from bot import gifs
from bot.attachments import AttachmentReader
from bot.discord_bot import ChatBot
from bot.gifs import GIF_FRAMES, download_gif, gif_embed, gif_source, looks_like_gif_link, prepare_gif
from bot.history import IMAGE_MARKER
from bot.images import ImageError

KLIPY_LINK = "https://klipy.com/gifs/no-no-no-dont-do-that-1"
# Aperçu tel que Discord l'envoie pour un GIF Klipy (relevé sur le bot de test).
KLIPY_EMBED = {
    "type": "gifv",
    "url": KLIPY_LINK,
    "title": "No No No Don'T Do That",
    "provider": {"name": "Klipy", "url": "https://klipy.com"},
    "thumbnail": {
        "url": "https://static.klipy.com/ii/a8ad/d9/67/Z2ghxPueS0WUci7G.webp",
        "proxy_url": "https://images-ext-1.discordapp.net/external/abc/https/static.klipy.com/ii/a8ad/d9/67/"
                     "Z2ghxPueS0WUci7G.webp",
        "width": 498, "height": 498,
    },
    "video": {"url": "https://static.klipy.com/ii/a8ad/d9/67/cxoiZ77HainGzQnl2.mp4", "width": 498, "height": 498},
}
GIPHY_EMBED = {
    "type": "gifv",
    "url": "https://giphy.com/gifs/dog-naughty-evil-smile-vNTIkYpoAUb22FhsVr",
    "provider": {"name": "Giphy", "url": "https://giphy.com"},
    "thumbnail": {
        "url": "https://i.giphy.com/media/vNTIkYpoAUb22FhsVr/giphy_s.gif",
        "proxy_url": "https://images-ext-1.discordapp.net/external/def/https/i.giphy.com/media/vNTIkYpoAUb22FhsVr/"
                     "giphy_s.gif",
    },
    "video": {"url": "https://i.giphy.com/media/vNTIkYpoAUb22FhsVr/giphy.mp4"},
}
# GIF des favoris de Discord : un lien vers un fichier gardé par Discord, avec un aperçu de type image.
FAVORITE_LINK = ("https://media.discordapp.net/attachments/411225883865513985/841065460211777576/pentadoigt.gif"
                 "?ex=6af1937f&is=6af041ff&hm=4f45c400d017a553ce3c9680260bd2b53863895185e6b4726c9d2a0470317962&")
FAVORITE_EMBED = {
    "type": "image",
    "url": FAVORITE_LINK,
    "thumbnail": {"url": FAVORITE_LINK, "proxy_url": FAVORITE_LINK, "width": 220, "height": 220},
}
TWEET_LINK = ("https://images-ext-1.discordapp.net/external/p2s2drywF0YN0p8Xum_7g2WeBHc0kp2UObSJ5zGJiag/https/"
              "gif.fxtwitter.com/tweet_video/HSM1wNAXYAAzOPT.webp?animated=true&width=495&height=541")
TWEET_EMBED = {
    "type": "image",
    "url": TWEET_LINK,
    "thumbnail": {"url": TWEET_LINK, "proxy_url": TWEET_LINK.replace("animated=true&", ""), "width": 495,
                  "height": 541},
}


def color(index: int) -> tuple[int, int, int]:
    return (index * 12, 0, 255 - index * 12)


def animation(count: int, fmt: str = "WEBP", size: tuple[int, int] = (100, 100)) -> bytes:
    """Animation de count images, chacune d'une couleur unie différente (voir color)."""
    frames = [Image.new("RGB", size, color(i)) for i in range(count)]
    buffer = io.BytesIO()
    frames[0].save(buffer, format=fmt, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buffer.getvalue()


def embed(data: dict) -> discord.Embed:
    return discord.Embed.from_dict(data)


class GifSourceTest(unittest.TestCase):
    def test_klipy_downloaded_animated_from_klipy(self):
        # La copie de Discord n'a que la première image : le GIF animé est pris chez Klipy.
        self.assertEqual(gif_source(embed(KLIPY_EMBED)), KLIPY_EMBED["thumbnail"]["url"])

    def test_other_sites_through_discord_copy(self):
        self.assertEqual(gif_source(embed(GIPHY_EMBED)), GIPHY_EMBED["thumbnail"]["proxy_url"])

    def test_nothing_downloaded_from_unknown_site(self):
        data = {**GIPHY_EMBED, "thumbnail": {"url": "https://exemple.org/a.gif",
                                             "proxy_url": "https://exemple.org/copie/a.gif"}}
        self.assertIsNone(gif_source(embed(data)))
        data["thumbnail"]["url"] = "http://static.klipy.com/a.webp"  # sans https
        self.assertIsNone(gif_source(embed(data)))

    def test_favorite_downloaded_from_its_own_link(self):
        # Le lien lui-même, gardé par Discord : il garde l'animation (animated=true).
        self.assertEqual(gif_source(embed(FAVORITE_EMBED)), FAVORITE_LINK)
        self.assertEqual(gif_source(embed(TWEET_EMBED)), TWEET_LINK)


class GifEmbedTest(unittest.TestCase):
    def test_favorites_recognized(self):
        for data in (FAVORITE_EMBED, TWEET_EMBED):
            with self.subTest(url=data["url"]):
                self.assertIsNotNone(gif_embed([embed(data)]))
                self.assertTrue(looks_like_gif_link(data["url"]))

    def test_image_from_other_site_ignored(self):
        # Une image d'un autre site n'est pas un GIF du sélecteur : elle n'est pas lue.
        data = {"type": "image", "url": "https://exemple.org/chat.gif",
                "thumbnail": {"url": "https://exemple.org/chat.gif",
                              "proxy_url": "https://images-ext-1.discordapp.net/external/x/https/exemple.org/chat.gif"}}
        self.assertIsNone(gif_embed([embed(data)]))
        self.assertFalse(looks_like_gif_link("https://exemple.org/chat.gif"))


class PrepareGifTest(unittest.TestCase):
    def assert_grid(self, data: bytes, count: int) -> None:
        image, shown = prepare_gif(data)
        self.assertEqual((image.media_type, shown), ("image/jpeg", GIF_FRAMES))
        grid = Image.open(io.BytesIO(image.data)).convert("RGB")
        gap = gifs.GRID_GAP
        self.assertEqual(grid.size, (3 * 100 + 2 * gap, 2 * 100 + gap))
        # Du début à la fin de l'animation, de gauche à droite puis de haut en bas.
        expected = [round(i * (count - 1) / (GIF_FRAMES - 1)) for i in range(GIF_FRAMES)]
        self.assertEqual(expected[0], 0)
        self.assertEqual(expected[-1], count - 1)
        for position, index in enumerate(expected):
            row, column = divmod(position, 3)
            pixel = grid.getpixel((column * (100 + gap) + 50, row * (100 + gap) + 50))
            for got, want in zip(pixel, color(index)):
                self.assertAlmostEqual(got, want, delta=12, msg=f"case {position} : image {index} attendue")

    def test_animated_webp_becomes_grid(self):
        self.assert_grid(animation(20), 20)

    def test_animated_gif_becomes_grid(self):
        self.assert_grid(animation(20, "GIF"), 20)

    def test_short_animation_shows_every_frame(self):
        image, shown = prepare_gif(animation(2))
        self.assertEqual(shown, 2)
        self.assertEqual(Image.open(io.BytesIO(image.data)).size, (2 * 100 + gifs.GRID_GAP, 100))

    def test_still_image_sent_alone(self):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100), "red").save(buffer, format="GIF")
        image, shown = prepare_gif(buffer.getvalue())
        self.assertEqual((image.media_type, shown), ("image/gif", 1))

    def test_cut_or_damaged_file_unreadable(self):
        # Fichier coupé en route ou abîmé : refusé comme illisible, jamais une autre erreur (qui ferait échouer
        # tout le message au lieu de le lire sans le GIF).
        noisy = [Image.frombytes("RGB", (60, 40), bytes((i * 37 + j) % 256 for j in range(60 * 40 * 3)))
                 for i in range(8)]
        for fmt in ("GIF", "WEBP"):
            buffer = io.BytesIO()
            noisy[0].save(buffer, format=fmt, save_all=True, append_images=noisy[1:], duration=100, loop=0)
            data = buffer.getvalue()
            for cut in range(10, len(data), 7):
                with self.subTest(fmt=fmt, cut=cut):
                    try:
                        prepare_gif(data[:cut])
                    except ImageError:
                        pass

    def test_huge_animation_not_played(self):
        # Des milliers d'images à décoder bloqueraient le bot : seule la première est envoyée.
        with mock.patch.object(gifs, "GIF_MAX_TOTAL_PIXELS", 100 * 100 * 10):
            image, shown = prepare_gif(animation(20))
        self.assertEqual(shown, 1)


class DownloadGifTest(unittest.IsolatedAsyncioTestCase):
    """Téléchargement réel, depuis un petit serveur local qui envoie le fichier en plusieurs morceaux."""

    async def serve(self, data: bytes, status: int = 200) -> str:
        async def handler(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(status=status)
            await response.prepare(request)
            for start in range(0, len(data), 1000):
                await response.write(data[start:start + 1000])
                await asyncio.sleep(0.01)  # le fichier arrive par morceaux, comme sur Internet
            await response.write_eof()
            return response
        app = web.Application()
        app.router.add_get("/gif", handler)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        return str(server.make_url("/gif"))

    DATA = bytes(range(256)) * 100  # 25,6 Ko, envoyés en 26 morceaux

    async def test_whole_file_downloaded(self):
        self.assertEqual(await download_gif(await self.serve(self.DATA)), self.DATA)

    async def test_too_heavy_refused(self):
        with mock.patch.object(gifs, "GIF_MAX_BYTES", 20_000):
            self.assertIsNone(await download_gif(await self.serve(self.DATA)))

    async def test_error_page_ignored(self):
        self.assertIsNone(await download_gif(await self.serve(b"introuvable", status=404)))


class ReadGifTest(unittest.IsolatedAsyncioTestCase):
    """Un GIF Klipy envoyé dans le salon dédié : Claude voit la grille, avec son explication."""

    def setUp(self):
        self.bot = bot = mock.Mock()
        bot.attachments = AttachmentReader(mock.Mock())
        bot.read_gif = ChatBot.read_gif.__get__(bot)
        bot.read_forward = mock.AsyncMock(return_value=(None, []))
        bot.read_attachments = mock.AsyncMock(return_value=([], []))
        bot.read_reply = mock.AsyncMock(return_value=(None, [], []))
        bot.message_body = ChatBot.message_body
        bot.timestamp.return_value = "07/11/2026 18:00"
        self.downloaded: list[str] = []

    async def download(self, url: str) -> bytes:
        self.downloaded.append(url)
        return animation(20)

    def message(self, embeds: list[discord.Embed]) -> mock.Mock:
        message = mock.Mock(id=1, attachments=[], content=KLIPY_LINK, clean_content=KLIPY_LINK, stickers=[],
                            embeds=embeds, created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))
        message.author.display_name = message.author.name = "membre"
        return message

    async def test_claude_sees_klipy_gif(self):
        message = self.message([embed(KLIPY_EMBED)])
        with mock.patch("bot.attachments.download_gif", self.download):
            pending = await ChatBot.read_main_message(self.bot, message, KLIPY_LINK)
        self.assertEqual(self.downloaded, [KLIPY_EMBED["thumbnail"]["url"]])
        self.assertEqual(len(pending.images), 1)
        self.assertEqual(pending.images[0].media_type, "image/jpeg")
        self.assertIn("GIF « No No No Don'T Do That » envoyé avec le sélecteur de Discord : l'image jointe montre "
                      "6 moments de l'animation", pending.user_text)
        self.assertEqual(pending.line["attachments"],
                         ["[GIF « No No No Don'T Do That » : tu l'as vu à ce moment-là, mais il n'est plus "
                          "conservé en mémoire]"])
        self.bot.read_reply.assert_awaited_once_with(message, want_images=False, text_budget=mock.ANY)

    async def test_claude_sees_favorite_gif(self):
        # GIF des favoris : son aperçu arrive aussi après le message, et il est attendu.
        message = self.message([])
        message.content = message.clean_content = FAVORITE_LINK

        async def preview_arrives():
            await asyncio.sleep(0.3)
            message.embeds = [embed(FAVORITE_EMBED)]
        arrival = asyncio.create_task(preview_arrives())
        with mock.patch("bot.attachments.download_gif", self.download):
            pending = await ChatBot.read_main_message(self.bot, message, FAVORITE_LINK)
        await arrival
        self.assertEqual(self.downloaded, [FAVORITE_LINK])
        self.assertEqual(len(pending.images), 1)
        self.assertIn("[GIF envoyé par lien : l'image jointe montre 6 moments de l'animation", pending.user_text)
        self.assertEqual(pending.line["attachments"],
                         ["[GIF : tu l'as vu à ce moment-là, mais il n'est plus conservé en mémoire]"])

    async def test_photo_shared_by_link_not_called_gif(self):
        # Une photo déjà postée sur Discord, partagée par son lien : Rika la voit, comme une image.
        link = "https://cdn.discordapp.com/attachments/1/2/vacances.jpg"
        content = f"regarde mes vacances {link}"
        message = self.message([embed({"type": "image", "url": link,
                                       "thumbnail": {"url": link, "proxy_url": link}})])
        message.content = message.clean_content = content
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100), "red").save(buffer, format="JPEG")
        with mock.patch("bot.attachments.download_gif", mock.AsyncMock(return_value=buffer.getvalue())):
            pending = await ChatBot.read_main_message(self.bot, message, content)
        self.assertEqual(len(pending.images), 1)
        self.assertIn("[Image envoyée par lien : elle est jointe à ce message]", pending.user_text)
        self.assertNotIn("GIF", pending.user_text)
        self.assertEqual(pending.line["attachments"], [IMAGE_MARKER])

    async def test_unreadable_gif_message_still_read(self):
        message = self.message([embed(KLIPY_EMBED)])
        with mock.patch("bot.attachments.download_gif", mock.AsyncMock(return_value=b"pas une image")):
            pending = await ChatBot.read_main_message(self.bot, message, KLIPY_LINK)
        self.assertEqual(pending.images, [])
        self.assertNotIn("attachments", pending.line)
        self.assertIn(KLIPY_LINK, pending.user_text)

    async def test_preview_arriving_after_message_awaited(self):
        # Discord ajoute l'aperçu un instant après le message : le GIF est quand même vu.
        message = self.message([])

        async def preview_arrives():
            await asyncio.sleep(0.3)
            message.embeds = [embed(KLIPY_EMBED)]
        arrival = asyncio.create_task(preview_arrives())
        with mock.patch("bot.attachments.download_gif", self.download):
            pending = await ChatBot.read_main_message(self.bot, message, KLIPY_LINK)
        await arrival
        self.assertEqual(len(pending.images), 1)
        self.assertIn("6 moments de l'animation", pending.user_text)

    async def test_preview_never_arriving_not_waited_forever(self):
        message = self.message([])
        with mock.patch("bot.discord_bot.EMBED_WAIT", 0.3), mock.patch("bot.attachments.download_gif", self.download):
            pending = await asyncio.wait_for(ChatBot.read_main_message(self.bot, message, KLIPY_LINK), 2)
        self.assertEqual((self.downloaded, pending.images), ([], []))

    async def test_ordinary_message_not_delayed(self):
        # Seul un lien de GIF attend son aperçu : un message ordinaire part sans attendre.
        # Un fichier gardé par Discord qui n'est pas une image n'aura pas d'aperçu : il n'est pas attendu non plus.
        for content in ("coucou", "regarde https://klipy.com/gifs/chat", "https://exemple.org/chat.gif",
                        "https://cdn.discordapp.com/attachments/1/2/archive.zip"):
            with self.subTest(content=content):
                message = self.message([])
                message.content = message.clean_content = content
                with mock.patch("bot.discord_bot.EMBED_WAIT", 30):
                    await asyncio.wait_for(ChatBot.read_main_message(self.bot, message, content), 1)

    async def test_plain_link_downloads_nothing(self):
        link = {"type": "link", "url": "https://exemple.org", "thumbnail": {"url": "https://exemple.org/a.png"}}
        message = self.message([embed(link)])
        with mock.patch("bot.attachments.download_gif", self.download):
            pending = await ChatBot.read_main_message(self.bot, message, KLIPY_LINK)
        self.assertEqual((self.downloaded, pending.images), ([], []))


if __name__ == "__main__":
    unittest.main()
