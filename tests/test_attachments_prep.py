"""Tests de la préparation des pièces jointes (images, PDF, fichiers texte). Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import codecs
import io
import random
import struct
import threading
import time
import unittest
import zlib
from types import SimpleNamespace
from unittest import mock

import aiohttp
from PIL import Image
from pypdf import PdfWriter

from bot import attachments, images, pdf_files
from bot.attachments import AttachmentReader
from bot.images import MAX_DIMENSION, MAX_PIXELS, ImageError, prepare_image
from bot.media import ImageData
from bot.pdf_files import PdfError, prepare_pdf
from bot.text_files import TextFileError, decode_text_file


def encode(img: Image.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def noise(size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    """Image de bruit (toujours le même) : presque impossible à compresser."""
    channels = len(mode)
    return Image.frombytes(mode, size, random.Random(0).randbytes(size[0] * size[1] * channels))


def opened(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def png_header(width: int, height: int) -> bytes:
    """PNG qui annonce ses dimensions mais ne contient aucun pixel."""
    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", b""))


class PrepareImageTest(unittest.TestCase):
    def test_small_jpeg_and_png_sent_as_is(self):
        for fmt, media_type in (("JPEG", "image/jpeg"), ("PNG", "image/png"),
                                ("GIF", "image/gif"), ("WEBP", "image/webp")):
            with self.subTest(fmt):
                data = encode(Image.new("RGB", (200, 100), "red"), fmt)
                result = prepare_image(data)
                self.assertEqual(result.media_type, media_type)
                self.assertIs(result.data, data)

    def test_exactly_max_dimension_sent_as_is(self):
        data = encode(Image.new("RGB", (MAX_DIMENSION, 100), "red"), "PNG")
        self.assertIs(prepare_image(data).data, data)

    def test_large_png_reduced(self):
        result = prepare_image(encode(Image.new("RGB", (3136, 1000), "blue"), "PNG"))
        self.assertEqual(result.media_type, "image/png")  # sans perte : le texte des captures reste net
        self.assertEqual(opened(result.data).size, (MAX_DIMENSION, 500))

    def test_large_jpeg_reduced(self):
        result = prepare_image(encode(Image.new("RGB", (1000, 3136), "green"), "JPEG"))
        self.assertEqual(result.media_type, "image/jpeg")
        self.assertEqual(opened(result.data).size, (500, MAX_DIMENSION))

    def test_unsupported_formats_converted(self):
        for fmt in ("BMP", "TIFF"):
            with self.subTest(fmt):
                result = prepare_image(encode(Image.new("RGB", (200, 100), "red"), fmt))
                self.assertEqual(result.media_type, "image/png")
                img = opened(result.data)
                self.assertEqual((img.format, img.size), ("PNG", (200, 100)))

    def test_high_bit_depth_grey_rescaled(self):
        # Gris 16 bits (scan, radio, astronomie) : ramené sur 8 bits, pas écrêté en blanc.
        cases = (("PNG", Image.new("I;16", (2000, 1000), 20000), 78),   # 20000/65535 ≈ 78/255
                 ("TIFF", Image.new("I", (200, 100), 20000), 78),       # entiers dans la plage 16 bits
                 ("TIFF", Image.new("F", (200, 100), 0.5), 128),        # flottants entre 0 et 1
                 ("TIFF", Image.new("F", (200, 100), 100.0), 100))      # déjà dans la plage 8 bits
        for fmt, img, expected in cases:
            with self.subTest(fmt=fmt, mode=img.mode, value=img.getpixel((0, 0))):
                result = opened(prepare_image(encode(img, fmt)).data)
                self.assertEqual(result.convert("L").getpixel((0, 0)), expected)

    def test_transparency_kept(self):
        result = prepare_image(encode(Image.new("RGBA", (3000, 100), (255, 0, 0, 0)), "PNG"))
        self.assertEqual(result.media_type, "image/png")
        self.assertEqual(opened(result.data).mode, "RGBA")

    def test_heavy_png_becomes_jpeg(self):
        # Limite de l'API abaissée : le PNG la dépasse, le JPEG (avec perte) passe en dessous.
        img = noise((300, 300))
        png, jpeg = encode(img, "PNG", optimize=True), encode(img, "JPEG", quality=45, optimize=True)
        self.assertLess(len(jpeg), len(png))
        with mock.patch.object(images, "API_MAX_BYTES", (len(png) + len(jpeg)) // 2):
            result = prepare_image(encode(img, "PNG"))
        self.assertEqual(result.media_type, "image/jpeg")
        self.assertLessEqual(len(result.data), (len(png) + len(jpeg)) // 2)

    def test_heavy_transparent_png_on_white(self):
        img = noise((300, 300), "RGBA")
        with mock.patch.object(images, "API_MAX_BYTES", len(encode(img, "PNG", optimize=True)) // 2):
            result = prepare_image(encode(img, "PNG"))
        self.assertEqual(result.media_type, "image/jpeg")
        self.assertEqual(opened(result.data).mode, "RGB")

    def test_too_heavy_even_compressed(self):
        with mock.patch.object(images, "API_MAX_BYTES", 1000), self.assertRaises(ImageError) as ctx:
            prepare_image(encode(noise((300, 300)), "PNG"))
        self.assertIn("compresser suffisamment", ctx.exception.user_message)

    def test_not_an_image(self):
        for data in (b"", b"bonjour, ceci n'est pas une image", encode(Image.new("RGB", (50, 50)), "PNG")[:40]):
            with self.subTest(data=data[:10]), self.assertRaises(ImageError) as ctx:
                prepare_image(data)
            self.assertIn("Je n'arrive pas à lire cette image", ctx.exception.user_message)

    def test_too_many_pixels_refused_before_decoding(self):
        # En-tête seul, sans pixels : refusée d'après l'en-tête, sinon le décodage échouerait.
        with self.assertRaises(ImageError) as ctx:
            prepare_image(png_header(MAX_PIXELS // 1000 + 1, 1000))
        self.assertIn("beaucoup trop grande", ctx.exception.user_message)
        with self.assertRaises(ImageError) as ctx:  # juste à la limite : décodage tenté
            prepare_image(png_header(MAX_PIXELS // 1000, 1000))
        self.assertIn("Je n'arrive pas à lire", ctx.exception.user_message)

    def test_pillow_limit_also_refused(self):
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000), self.assertRaises(ImageError) as ctx:
            prepare_image(encode(Image.new("RGB", (100, 100)), "PNG"))
        self.assertIn("beaucoup trop grande", ctx.exception.user_message)

    def test_baseline_jpeg_allowed_more_pixels(self):
        # Seuils abaissés pour ne pas fabriquer une vraie image géante.
        img = Image.new("RGB", (3136, 1000), "green")
        with mock.patch.object(images, "MAX_PIXELS", 1000):
            self.assertEqual(opened(prepare_image(encode(img, "JPEG")).data).size, (MAX_DIMENSION, 500))
            with self.assertRaises(ImageError) as ctx:  # progressif : gardé en entier au décodage
                prepare_image(encode(img, "JPEG", progressive=True))
            self.assertIn("beaucoup trop grande", ctx.exception.user_message)
        with mock.patch.object(images, "MAX_JPEG_PIXELS", 1000), self.assertRaises(ImageError):
            prepare_image(encode(img, "JPEG"))

    def test_jpeg_decoded_small(self):
        # Panorama : réduit au décodage d'après son grand côté, pas son petit.
        decoded_sizes = []
        thumbnail = Image.Image.thumbnail

        def spy(img, *args, **kwargs):
            decoded_sizes.append(img.size)
            return thumbnail(img, *args, **kwargs)

        with mock.patch.object(Image.Image, "thumbnail", autospec=True, side_effect=spy):
            result = prepare_image(encode(Image.new("RGB", (4 * MAX_DIMENSION, 1000), "green"), "JPEG"))
        self.assertEqual(decoded_sizes, [(MAX_DIMENSION, 250)])
        self.assertEqual(opened(result.data).size, (MAX_DIMENSION, 250))

    def test_exif_orientation_applied(self):
        img = Image.new("RGB", (3136, 1000), "green")
        exif = img.getexif()
        exif[0x0112] = 6  # photo prise téléphone tourné : à pivoter d'un quart de tour
        result = prepare_image(encode(img, "JPEG", exif=exif))
        self.assertEqual(opened(result.data).size, (500, MAX_DIMENSION))


class ImagesAtOnceTest(unittest.IsolatedAsyncioTestCase):
    async def test_images_prepared_one_at_a_time(self):
        running = peak = 0
        lock = threading.Lock()

        def slow_prepare(data: bytes) -> ImageData:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with lock:
                running -= 1
            return ImageData("image/png", data)

        reader, files = self.reader_and_files(4)
        with mock.patch.object(attachments, "prepare_image", slow_prepare):
            results = await asyncio.gather(reader.download_images(files[:2]),
                                           reader.quoted_images(files[2:], "cité"))
        self.assertEqual(peak, attachments.MAX_IMAGES_AT_ONCE)  # tous messages confondus
        self.assertEqual([r.data for batch in results for r in batch], [b"\0", b"\1", b"\2", b"\3"])

    async def test_slot_kept_until_cancelled_image_is_decoded(self):
        # Annuler l'attente n'arrête pas le thread : sa place reste prise jusqu'à ce qu'il finisse.
        started = []
        release = threading.Event()

        def blocking_prepare(data: bytes) -> ImageData:
            started.append(data)
            release.wait(5)
            return ImageData("image/png", data)

        reader, files = self.reader_and_files(2)
        with mock.patch.object(attachments, "prepare_image", blocking_prepare):
            try:
                first = asyncio.create_task(reader.download_image(files[0]))
                while not started:
                    await asyncio.sleep(0.01)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(reader.download_image(files[1]))
                await asyncio.sleep(0.1)
                self.assertEqual(started, [b"\0"])  # la première décode encore : la deuxième attend
            finally:
                release.set()
            self.assertEqual((await second).data, b"\1")

    @staticmethod
    def reader_and_files(count: int) -> tuple[AttachmentReader, list[SimpleNamespace]]:
        reader = AttachmentReader(SimpleNamespace(max_image_size=10_000_000))
        files = [SimpleNamespace(filename=f"{i}.png", content_type="image/png", size=10,
                                 read=mock.AsyncMock(return_value=bytes([i])))
                 for i in range(count)]
        return reader, files


class DownloadFailureTest(unittest.IsolatedAsyncioTestCase):
    """Une coupure réseau ou un délai dépassé pendant le téléchargement prévient le membre au lieu de tout faire échouer."""

    def setUp(self):
        self.reader = AttachmentReader(SimpleNamespace(max_image_size=10_000_000, pdf_max_pages=10))

    @staticmethod
    def file(name: str, read) -> SimpleNamespace:
        return SimpleNamespace(filename=name, content_type=None, size=10, read=read)

    def broken(self, name: str) -> SimpleNamespace:
        return self.file(name, mock.AsyncMock(side_effect=aiohttp.ClientConnectionError("connexion réinitialisée")))

    async def test_connection_error_asks_to_retry(self):
        for name, error in (("photo.png", ImageError), ("doc.pdf", PdfError), ("notes.txt", TextFileError)):
            with self.subTest(name), self.assertLogs("chatbot.discord", "ERROR"):
                with self.assertRaises(error) as caught:
                    await self.reader.read([self.broken(name)])
                self.assertIn("réessaie", caught.exception.user_message)

    async def test_quoted_image_unreadable_is_ignored(self):
        good = self.file("ok.png", mock.AsyncMock(return_value=b"ok"))
        with mock.patch.object(attachments, "prepare_image", lambda data: ImageData("image/png", data)), \
                self.assertLogs("chatbot.discord", "WARNING"):
            result = await self.reader.quoted_images([self.broken("ko.png"), good], "cité")
        self.assertEqual([r.data for r in result], [b"ok"])

    async def test_download_timeout(self):
        async def never_ends():
            await asyncio.sleep(10)

        with mock.patch.object(attachments, "DOWNLOAD_TIMEOUT", 0.01), self.assertLogs("chatbot.discord", "ERROR"):
            with self.assertRaises(ImageError):
                await self.reader.download_image(self.file("lente.png", never_ends))


def pdf(pages: int, password: str | None = None, owner_password: str | None = None,
        algorithm: str | None = None) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    if password is not None:
        writer.encrypt(user_password=password, owner_password=owner_password or "propriétaire", algorithm=algorithm)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class PreparePdfTest(unittest.TestCase):
    def test_valid_pdf(self):
        data = pdf(3)
        result = prepare_pdf("cours.pdf", data, max_pages=3)
        self.assertTrue(result.is_pdf)
        self.assertEqual(result.name, "cours.pdf")
        self.assertIs(result.data, data)
        self.assertEqual(result.pages, 3)  # compte dans le budget de la requête (voir media.fit_in_request)

    def test_too_many_pages(self):
        with self.assertRaises(PdfError) as ctx:
            prepare_pdf("cours.pdf", pdf(4), max_pages=3)
        self.assertEqual(ctx.exception.user_message,
                         "Ce PDF a 4 pages : je n'en lis que 3 au maximum. Envoie-moi seulement les pages utiles.")

    def test_no_page(self):
        with self.assertRaises(PdfError) as ctx:
            prepare_pdf("vide.pdf", pdf(0), max_pages=3)
        self.assertIn("aucune page", ctx.exception.user_message)

    def test_password(self):
        with self.assertRaises(PdfError) as ctx:
            prepare_pdf("secret.pdf", pdf(1, password="secret"), max_pages=3)
        self.assertIn("mot de passe", ctx.exception.user_message)

    def test_owner_password_only(self):
        # Chiffré sans mot de passe d'ouverture (seulement des droits limités) : lisible.
        self.assertTrue(prepare_pdf("protégé.pdf", pdf(1, password=""), max_pages=3).is_pdf)

    def test_owner_password_only_aes(self):
        # Chiffrement AES-256 (Acrobat, LibreOffice récents) : pypdf a besoin de la bibliothèque cryptography.
        data = pdf(2, password="", owner_password="propriétaire", algorithm="AES-256")
        self.assertEqual(prepare_pdf("protégé.pdf", data, max_pages=3).pages, 2)

    def test_missing_decryption_library(self):
        # Sans bibliothèque de déchiffrement, pypdf refuse le PDF : le dire clairement, et le signaler à l'administrateur.
        error = pdf_files.DependencyError("cryptography>=3.1 is required for AES algorithm")
        with mock.patch.object(pdf_files, "PdfReader", side_effect=error), \
                self.assertLogs("chatbot.pdf", "WARNING") as logs, self.assertRaises(PdfError) as ctx:
            prepare_pdf("protégé.pdf", b"%PDF", max_pages=3)
        self.assertIn("protégé", ctx.exception.user_message)
        self.assertIn("cryptography", logs.output[0])

    def test_missing_other_library(self):
        # Une autre bibliothèque manquante (brotli : compression rare) ne fait pas croire à un PDF protégé.
        error = pdf_files.DependencyError("brotli is required for BrotliDecode. Install it with: pip install pypdf[brotli]")
        with mock.patch.object(pdf_files, "PdfReader", side_effect=error), \
                self.assertLogs("chatbot.pdf", "WARNING") as logs, self.assertRaises(PdfError) as ctx:
            prepare_pdf("compressé.pdf", b"%PDF", max_pages=3)
        self.assertNotIn("protégé", ctx.exception.user_message)
        self.assertIn("brotli", logs.output[0])

    def test_not_a_pdf(self):
        for data in (b"", b"bonjour, ceci n'est pas un PDF", encode(Image.new("RGB", (50, 50)), "PNG")):
            with self.subTest(data=data[:10]), self.assertLogs("chatbot.pdf", "INFO"), \
                    self.assertRaises(PdfError) as ctx:
                prepare_pdf("faux.pdf", data, max_pages=3)
            self.assertEqual(ctx.exception.user_message, "Je n'arrive pas à lire ce PDF.")


class DecodeTextFileTest(unittest.TestCase):
    def test_utf8(self):
        self.assertEqual(decode_text_file("Été à Noël 😀".encode("utf-8")), "Été à Noël 😀")

    def test_utf8_with_bom(self):
        self.assertEqual(decode_text_file(codecs.BOM_UTF8 + "Été".encode("utf-8")), "Été")

    def test_windows_1252(self):
        # « ANSI » du Bloc-notes : pas de l'UTF-8 valide, lu en Windows-1252.
        self.assertEqual(decode_text_file("Été, 5 € œuvre".encode("cp1252")), "Été, 5 € œuvre")

    def test_bytes_undefined_in_windows_1252(self):
        self.assertEqual(decode_text_file(b"caf\xe9 \x81"), "café �")

    def test_utf16_with_bom(self):
        for encoding in ("utf-16-le", "utf-16-be"):
            with self.subTest(encoding):
                bom = codecs.BOM_UTF16_LE if encoding == "utf-16-le" else codecs.BOM_UTF16_BE
                self.assertEqual(decode_text_file(bom + "Été".encode(encoding)), "Été")

    def test_broken_utf16(self):
        with self.assertRaises(TextFileError) as ctx:
            decode_text_file(codecs.BOM_UTF16_LE + b"a\x00b")  # nombre impair d'octets
        self.assertIn("pas du texte lisible", ctx.exception.user_message)

    def test_line_endings_and_spaces(self):
        self.assertEqual(decode_text_file(b"\r\n  ligne 1\r\nligne 2\r\n\r\n"), "ligne 1\nligne 2")

    def test_empty(self):
        for data in (b"", b"  \r\n\t ", codecs.BOM_UTF8):
            with self.subTest(data=data), self.assertRaises(TextFileError) as ctx:
                decode_text_file(data)
            self.assertIn("vide", ctx.exception.user_message)

    def test_binary(self):
        for data in (bytes(range(256)), encode(Image.new("RGB", (50, 50)), "PNG"), "texte".encode("utf-16-le")):
            with self.subTest(data=data[:10]), self.assertRaises(TextFileError) as ctx:
                decode_text_file(data)
            self.assertIn("pas du texte lisible", ctx.exception.user_message)


if __name__ == "__main__":
    unittest.main()
