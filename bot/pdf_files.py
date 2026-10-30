"""Lecture des PDF joints aux messages : envoyés tels quels à Claude, qui en lit le texte et les pages."""

from __future__ import annotations

import io
import logging

from pypdf import PdfReader

from .claude_client import PDF_MEDIA_TYPE, ImageData

logger = logging.getLogger("chatbot.pdf")
logging.getLogger("pypdf").setLevel(logging.ERROR)  # avertissements sur les PDF mal formés, sans gravité

PDF_EXTENSIONS = (".pdf",)
# Limite de l'API : 32 Mo par requête, une fois encodé en base64 (+33 %), historique compris.
PDF_MAX_BYTES = 20_000_000


class PdfError(Exception):
    """PDF illisible ou refusé, avec un message affichable sur Discord."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


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
    except Exception as exc:  # fichier abîmé ou pas un PDF : pypdf lève des erreurs très variées
        logger.info("PDF %s illisible : %s", name, exc)
        raise PdfError("Je n'arrive pas à lire ce PDF.")
    if pages == 0:
        raise PdfError("Ce PDF ne contient aucune page.")
    if pages > max_pages:
        raise PdfError(f"Ce PDF a {pages} pages : je n'en lis que {max_pages} au maximum. "
                       "Envoie-moi seulement les pages utiles.")
    logger.info("PDF %s : %d page(s), %.1f Mo", name, pages, len(data) / 1e6)
    return ImageData(PDF_MEDIA_TYPE, data, name=name)
