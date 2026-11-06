"""Pièces jointes préparées pour Claude (images et PDF), et ce qu'elles pèsent dans une requête."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest

PDF_MEDIA_TYPE = "application/pdf"

# Budget de pièces jointes par requête. Les messages arrivés pendant une réponse partent ensemble :
# sans budget, plusieurs gros fichiers dépasseraient les limites de l'API (128 Mo par requête ; 100 pages
# de PDF et 100 images pour les modèles à 200 000 tokens de contexte) ou sa fenêtre de contexte, et
# occuperaient beaucoup de mémoire le temps de l'envoi.
REQUEST_MAX_BASE64 = 24_000_000  # octets une fois encodés en base64 (+33 %), soit 18 Mo bruts
REQUEST_MAX_PAGES = 100  # pages de PDF et images, jusqu'à plusieurs milliers de tokens chacune
# Caractères de texte par requête, fichiers texte joints compris : environ 100 000 tokens, soit trois fichiers
# à leur taille maximale (TEXT_FILE_MAX_BYTES). Laisse de la place à l'historique, et limite le coût d'un appel.
REQUEST_MAX_TEXT = 350_000


@dataclass
class ImageData:
    """Image ou PDF joint à un message, prêt à être envoyé à Claude."""
    media_type: str
    data: bytes
    name: str | None = None  # nom du fichier (PDF)
    pages: int = 1  # pages d'un PDF ; une image compte pour une page

    @property
    def is_pdf(self) -> bool:
        return self.media_type == PDF_MEDIA_TYPE


def base64_size(size: int) -> int:
    """Taille de size octets une fois encodés en base64."""
    return 4 * -(-size // 3)


def request_weight(attachments: list[ImageData]) -> tuple[int, int]:
    """Poids de pièces jointes dans une requête : (octets en base64, pages de PDF et images)."""
    return sum(base64_size(len(a.data)) for a in attachments), sum(a.pages for a in attachments)


def fit_in_request(messages: list[list[ImageData]], texts: Sequence[str] = ()) -> int:
    """Nombre de messages, pris dans l'ordre, qui partent ensemble dans la prochaine requête : tant que
    leurs pièces jointes (une liste par message) et leurs textes tiennent dans le budget. Le premier part
    toujours, même seul au-delà du budget : il tient déjà dans les limites de l'API (voir PDF_MAX_BYTES,
    MAX_IMAGES, TEXT_FILE_MAX_BYTES)."""
    size = pages = characters = 0
    for count, (attachments, text) in enumerate(zip_longest(messages, texts, fillvalue="")):
        message_size, message_pages = request_weight(attachments)
        size, pages, characters = size + message_size, pages + message_pages, characters + len(text)
        if count and (size > REQUEST_MAX_BASE64 or pages > REQUEST_MAX_PAGES or characters > REQUEST_MAX_TEXT):
            return count
    return len(messages)
