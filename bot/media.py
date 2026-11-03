"""Pièces jointes préparées pour Claude (images et PDF)."""

from __future__ import annotations

from dataclasses import dataclass

PDF_MEDIA_TYPE = "application/pdf"


@dataclass
class ImageData:
    """Image ou PDF joint à un message, prêt à être envoyé à Claude."""
    media_type: str
    data: bytes
    name: str | None = None  # nom du fichier (PDF)

    @property
    def is_pdf(self) -> bool:
        return self.media_type == PDF_MEDIA_TYPE
