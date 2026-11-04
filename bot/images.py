"""Préparation des images avant l'envoi à Claude (redimensionnement et compression)."""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageFilter, ImageMath, ImageOps, UnidentifiedImageError

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
# Image en flottants : une valeur infinie, NaN ou plus grande que ceci (en valeur absolue) n'est pas une
# mesure mais un trou (zone « sans donnée » d'une carte, souvent ±3,4e38). Assez haut pour garder les vraies
# valeurs géantes (densités de colonne en astronomie, autour de 1e21).
_FLOAT_HOLE = 1e30
_FLOAT32_MAX = 3.4e38
# Recherche des trous par bandes d'environ ce nombre de pixels : jamais une copie entière d'une grande image.
_STRIP_PIXELS = 1_000_000


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
    holes = _fill_float_holes(img) if img.mode == "F" else None
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
    if holes is not None:
        _restore_float_holes(img, holes.resize(img.size, Image.Resampling.BOX))
    # Après la réduction : la copie éventuelle est petite. Respecte l'orientation des photos de téléphone.
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I") or img.mode == "F":
        img = _to_8_bits(img)

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


def _float_measures(img: Image.Image) -> Image.Image:
    """Masque d'une image en flottants : 255 pour une mesure, 0 pour un trou (NaN : la comparaison est fausse).

    Calculé par Pillow : lire les pixels un par un en Python serait lent et gourmand en mémoire.
    """
    return ImageMath.lambda_eval(lambda a: (abs(a["img"]) < _FLOAT_HOLE) * 255, img=img).convert("L")


def _fill_float_holes(img: Image.Image) -> Image.Image | None:
    """Bouche les trous d'une image en flottants avant sa réduction, et renvoie leur masque (255 = trou).

    Réduits tels quels, les trous laisseraient autour d'eux des traces (lobes du filtre LANCZOS) de toutes
    tailles, jusqu'à quelques dizaines : impossibles à distinguer des mesures, elles fausseraient les gris.
    Ils sont remplacés par la plus petite mesure (bord sombre, comme le noir des trous). Parcours par bandes
    de lignes : chaque calcul copie la bande, et une copie entière d'une grande image pèserait des centaines
    de Mo. Renvoie None s'il n'y a aucun trou (l'image n'est alors pas modifiée).
    """
    holes = None
    lowest = None
    rows = max(1, _STRIP_PIXELS // img.width)
    for top in range(0, img.height, rows):
        strip = img.crop((0, top, img.width, min(top + rows, img.height)))
        measures = _float_measures(strip)
        fewest, most = measures.getextrema()
        if fewest < 255:  # des trous dans cette bande
            if holes is None:
                holes = Image.new("L", img.size, 0)
            strip_holes = ImageOps.invert(measures)
            holes.paste(strip_holes, (0, top))
            if most == 0:  # que des trous
                continue
            strip.paste(_FLOAT32_MAX, mask=strip_holes)  # ignorés pour le minimum
        strip_low = strip.getextrema()[0]
        lowest = strip_low if lowest is None else min(lowest, strip_low)
    if holes is not None:
        img.paste(0.0 if lowest is None else lowest, mask=holes)
    return holes


def _restore_float_holes(img: Image.Image, holes: Image.Image) -> None:
    """Remet dans l'image réduite les trous bouchés par _fill_float_holes, en NaN (_to_8_bits les rend noirs).

    `holes` : leur masque réduit à la taille de l'image, par moyenne (part de trous de chaque pixel, sans
    rebond) ; un pixel surtout fait de trous en redevient un. Autour, sur la portée du filtre LANCZOS
    (3 pixels), les valeurs mêlent le bouchage aux mesures et peuvent déborder : elles sont ramenées dans la
    plage des pixels plus loin des trous, pour ne pas fausser l'échelle des gris.
    """
    # À 4 pixels ou moins d'un trou (moyenne sur 9 × 9 non nulle : plus rapide qu'un MaxFilter).
    near = holes.point(lambda v: 255 if v else 0).filter(ImageFilter.BoxBlur(4)).point(lambda v: 255 if v else 0)
    low_side, high_side = img.copy(), img.copy()  # bornes des pixels loin des trous (voir _to_8_bits)
    low_side.paste(_FLOAT32_MAX, mask=near)
    high_side.paste(-_FLOAT32_MAX, mask=near)
    low, high = low_side.getextrema()[0], high_side.getextrema()[1]
    if low <= high:  # sinon aucun pixel loin des trous : rien à quoi se ramener
        img.paste(ImageMath.lambda_eval(lambda a: a["min"](a["max"](a["img"], low), high), img=img), mask=near)
    img.paste(float("nan"), mask=holes.point(lambda v: 255 if v > 127 else 0))


def _to_8_bits(img: Image.Image) -> Image.Image:
    """Ramène une image en gris 16 bits, entière ou flottante (scans, radios, astronomie) sur 8 bits.

    Les conversions de Pillow écrêtent au lieu de remettre à l'échelle : tout ce qui dépasse 255
    deviendrait blanc. L'échelle dépend des valeurs présentes : 0–255 gardé tel quel, 0–1 (flottants),
    0–65 535 (entiers 16 bits), sinon de la plus petite à la plus grande valeur ; gris moyen si l'image
    est uniforme. Les trous d'une image flottante (voir _FLOAT_HOLE) ne comptent pas et deviennent noirs.
    """
    holes = None
    if img.mode.startswith("I;16"):
        low, high = 0, 65535
    else:
        if img.mode == "F":
            measures = _float_measures(img)
            fewest, most = measures.getextrema()
            if fewest == 255:  # aucun trou
                low, high = img.getextrema()
            elif most == 0:  # que des trous : tout noir
                holes = ImageOps.invert(measures)
                low, high = 0, 0
            else:
                holes = ImageOps.invert(measures)
                # Bornes des seules mesures : les trous remplacés par le plus grand flottant pour le
                # minimum, par le plus petit pour le maximum.
                low_side, high_side = img.copy(), img.copy()
                low_side.paste(_FLOAT32_MAX, mask=holes)
                high_side.paste(-_FLOAT32_MAX, mask=holes)
                low, high = low_side.getextrema()[0], high_side.getextrema()[1]
        else:
            low, high = img.getextrema()
        if low >= 0 and high <= 1:
            low, high = 0, 1
        elif low >= 0 and high <= 255:
            low, high = 0, 255
        elif low >= 0 and high <= 65535 and img.mode != "F":  # flottants : mètres, degrés…, pas 16 bits
            low, high = 0, 65535
        elif high == low:  # image uniforme hors de ces plages : gris moyen, pas noir
            # Écart proportionnel à la valeur : ajouter 1 à un très grand flottant ne le change plus.
            margin = max(1.0, abs(low))
            low, high = low - margin, high + margin
    img = img.convert("F")
    scale = 255 / (high - low)
    result = img.point(lambda v: (v - low) * scale + 0.5).convert("L")  # + 0.5 : arrondi, pas troncature
    if holes:
        result.paste(0, mask=holes)
    return result


def _encode(img: Image.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **options)
    return buffer.getvalue()
