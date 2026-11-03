"""Préparation des images avant l'envoi à Claude (redimensionnement et compression)."""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import UserFacingError
from .media import ImageData

logger = logging.getLogger("chatbot.images")

# Claude redimensionne lui-même toute image dont le grand côté dépasse ~1568 px :
# envoyer plus grand ne lui apporte rien.
MAX_DIMENSION = 1568
# Limite de l'API : 5 Mo une fois encodé en base64, soit environ 3,75 Mo bruts.
API_MAX_BYTES = 3_750_000
# Protection contre les images piégées : un petit fichier peut cacher des centaines de millions
# de pixels, qui saturent la mémoire une fois décodés. Refusées dès la lecture de l'en-tête,
# avant tout décodage (Pillow, lui, ne refuse qu'au double de sa limite et avertit seulement entre les deux).
MAX_PIXELS = 40_000_000
# JPEG classique (non progressif) : décodé directement en petit, il coûte peu même très grand
# (photos de 50 à 108 Mpx des téléphones récents). Un JPEG progressif, lui, est gardé en entier.
MAX_JPEG_PIXELS = 120_000_000
Image.MAX_IMAGE_PIXELS = MAX_JPEG_PIXELS

_DIRECT_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
# Formats sans perte (captures d'écran, dessins…) : PNG essayé avant le JPEG.
_LOSSLESS_FORMATS = {"PNG", "GIF", "BMP", "TIFF"}
_PNG_MODES = ("L", "RGB")  # modes sans transparence enregistrables tels quels en PNG


class ImageError(UserFacingError):
    """Image illisible ou impossible à préparer, avec un message affichable sur Discord."""


def prepare_image(data: bytes) -> ImageData:
    """Renvoie une image acceptée par l'API, en la réduisant ou la convertissant si besoin.

    Fonction bloquante (calculs sur l'image) : à appeler via asyncio.to_thread.
    """
    try:
        img = Image.open(io.BytesIO(data))  # ne lit que l'en-tête
        fmt = (img.format or "").upper()
        baseline_jpeg = fmt in ("JPEG", "MPO") and not img.info.get("progressive")
        if img.width * img.height > (MAX_JPEG_PIXELS if baseline_jpeg else MAX_PIXELS):
            raise Image.DecompressionBombError
        as_is = fmt in _DIRECT_FORMATS and len(data) <= API_MAX_BYTES and max(img.size) <= MAX_DIMENSION
        original_size, original_dims = len(data), img.size
        if not as_is:
            # JPEG : décodé directement à 1/2, 1/4 ou 1/8 de sa taille, sans passer sous la taille
            # finale (sans effet sur les autres formats).
            ratio = MAX_DIMENSION / max(img.size)
            img.draft("RGB", (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))
        img.load()
    except Image.DecompressionBombError:
        raise ImageError("Cette image est beaucoup trop grande (en pixels) pour que je puisse la lire.")
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageError("Je n'arrive pas à lire cette image. Formats conseillés : JPEG, PNG, GIF ou WebP.")

    if as_is:
        return ImageData(media_type=_DIRECT_FORMATS[fmt], data=data)  # déjà parfaite : envoyée telle quelle

    img.seek(0)                      # GIF ou WebP animé : première image uniquement
    if img.mode in ("1", "P"):  # sinon réduite au plus proche voisin, qui abîme le texte
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
    # Après la réduction : la copie éventuelle est petite. Respecte l'orientation des photos de téléphone.
    img = ImageOps.exif_transpose(img)

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    result: ImageData | None = None

    # PNG si la taille le permet : garde la transparence, et le texte net des captures d'écran
    # (le JPEG le brouille). Les photos, déjà en JPEG, passent directement au JPEG.
    if has_alpha or fmt in _LOSSLESS_FORMATS:
        if has_alpha:
            png = _encode(img.convert("RGBA"), "PNG", optimize=True)
        else:
            png = _encode(img if img.mode in _PNG_MODES else img.convert("RGB"), "PNG", optimize=True)
        if len(png) <= API_MAX_BYTES:
            result = ImageData("image/png", png)
        elif has_alpha:  # sinon fond blanc + JPEG
            background = Image.new("RGB", img.size, "white")
            background.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
            img = background

    if result is None:
        rgb = img.convert("RGB")
        for quality in (85, 75, 60, 45):
            jpeg = _encode(rgb, "JPEG", quality=quality, optimize=True)
            if len(jpeg) <= API_MAX_BYTES:
                result = ImageData("image/jpeg", jpeg)
                break
        else:
            raise ImageError("Je n'ai pas réussi à compresser suffisamment cette image.")

    logger.info("Image adaptée : %s %dx%d (%.1f Mo) → %s %dx%d (%.1f Mo)",
                fmt or "?", *original_dims, original_size / 1e6,
                result.media_type, *img.size, len(result.data) / 1e6)
    return result


def _encode(img: Image.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **options)
    return buffer.getvalue()
