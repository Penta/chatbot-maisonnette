"""Lecture des PDF joints aux messages : envoyés tels quels à Claude, qui en lit le texte et les pages."""

from __future__ import annotations

import io
import logging

from pypdf import PdfReader
from pypdf.errors import DependencyError

from .errors import UserFacingError
from .media import PDF_MEDIA_TYPE, ImageData

logger = logging.getLogger("chatbot.pdf")
logging.getLogger("pypdf").setLevel(logging.ERROR)  # avertissements sur les PDF mal formés, sans gravité

PDF_EXTENSIONS = (".pdf",)
# 26,7 Mo une fois encodé en base64 (+33 %) : bien en dessous de la limite de l'API (128 Mo par requête).
# Plusieurs PDF arrivés pendant une réponse partent au besoin dans des requêtes séparées (voir
# media.REQUEST_MAX_BASE64) ; seul, un PDF peut dépasser ce budget.
PDF_MAX_BYTES = 20_000_000


class PdfError(UserFacingError):
    """PDF illisible ou refusé, avec un message affichable sur Discord."""


def prepare_pdf(name: str, data: bytes, max_pages: int) -> ImageData:
    """Vérifie un PDF avant l'envoi à Claude : lisible, sans mot de passe, max_pages pages au plus.

    Fonction bloquante (lecture du PDF) : à appeler via asyncio.to_thread.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfError("Ce PDF est protégé par un mot de passe, je ne peux pas le lire.")
        pages = len(reader.pages)
    except PdfError:
        raise
    except DependencyError as exc:  # bibliothèque manquante sur le serveur : à corriger par l'administrateur
        logger.warning("PDF %s illisible faute de bibliothèque : %s", name, exc)
        if "cryptography" in str(exc):  # chiffrement AES-256 ; brotli (compression rare) lève la même erreur
            raise PdfError("Ce PDF est protégé et je ne peux pas l'ouvrir pour le moment.")
        raise PdfError("Je ne peux pas lire ce PDF pour le moment.")
    except Exception as exc:  # fichier abîmé ou pas un PDF : pypdf lève des erreurs très variées
        logger.info("PDF %s illisible : %s", name, exc)
        raise PdfError("Je n'arrive pas à lire ce PDF.")
    if pages == 0:
        raise PdfError("Ce PDF ne contient aucune page.")
    if pages > max_pages:
        raise PdfError(f"Ce PDF a {pages} pages : je n'en lis que {max_pages} au maximum. "
                       "Envoie-moi seulement les pages utiles.")
    logger.info("PDF %s : %d page(s), %.1f Mo", name, pages, len(data) / 1e6)
    return ImageData(PDF_MEDIA_TYPE, data, name=name, pages=pages)
