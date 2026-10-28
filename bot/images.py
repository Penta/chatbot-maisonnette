"""Préparation des images avant l'envoi à Claude (redimensionnement et compression)."""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageOps, UnidentifiedImageError

from .claude_client import ImageData

logger = logging.getLogger("chatbot.images")

# Claude redimensionne lui-même toute image dont le grand côté dépasse ~1568 px :
# envoyer plus grand ne lui apporte rien.
MAX_DIMENSION = 1568
# Limite de l'API : 5 Mo une fois encodé en base64, soit environ 3,75 Mo bruts.
API_MAX_BYTES = 3_750_000
# Protection contre les images piégées (des milliards de pixels qui saturent la mémoire).
Image.MAX_IMAGE_PIXELS = 80_000_000

_DIRECT_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}


class ImageError(Exception):
    """Image illisible ou impossible à préparer, avec un message affichable sur Discord."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def prepare_image(data: bytes) -> ImageData:
    """Renvoie une image acceptée par l'API, en la réduisant ou la convertissant si besoin.

    Fonction bloquante (calculs sur l'image) : à appeler via asyncio.to_thread.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        raise ImageError("Cette image est beaucoup trop grande (en pixels) pour que je puisse la lire.")
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageError("Je n'arrive pas à lire cette image. Formats conseillés : JPEG, PNG, GIF ou WebP.")

    fmt = (img.format or "").upper()
    if fmt in _DIRECT_FORMATS and len(data) <= API_MAX_BYTES and max(img.size) <= MAX_DIMENSION:
        return ImageData(media_type=_DIRECT_FORMATS[fmt], data=data)  # déjà parfaite : envoyée telle quelle

    original_size, original_dims = len(data), img.size
    img.seek(0)                      # GIF ou WebP animé : première image uniquement
    img = ImageOps.exif_transpose(img)  # respecte l'orientation des photos de téléphone
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    result: ImageData | None = None

    if has_alpha:  # garde la transparence en PNG si la taille le permet
        png = _encode(img.convert("RGBA"), "PNG", optimize=True)
        if len(png) <= API_MAX_BYTES:
            result = ImageData("image/png", png)
        else:  # sinon fond blanc + JPEG
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
