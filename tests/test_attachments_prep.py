"""Tests de la préparation des pièces jointes (images, PDF, fichiers texte). Lancer depuis la racine du projet : python -m unittest"""

import codecs
import io
import random
import unittest
from unittest import mock

from PIL import Image
from pypdf import PdfWriter

from bot import images
from bot.images import MAX_DIMENSION, ImageError, prepare_image
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

    def test_too_many_pixels(self):
        # Seuil abaissé pour ne pas fabriquer une vraie image géante.
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000), self.assertRaises(ImageError) as ctx:
            prepare_image(encode(Image.new("RGB", (100, 100)), "PNG"))
        self.assertIn("beaucoup trop grande", ctx.exception.user_message)


def pdf(pages: int, password: str | None = None, owner_password: str | None = None) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    if password is not None:
        writer.encrypt(user_password=password, owner_password=owner_password or "propriétaire")
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
