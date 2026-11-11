"""Pièces jointes des messages : reconnaissance, vérification et téléchargement."""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

import aiohttp
import discord

from .config import Settings
from .errors import UserFacingError
from .gifs import download_gif, from_picker, gif_embed, gif_marker, gif_prompt, gif_source, gif_title, prepare_gif
from .history import images_marker
from .images import IMAGE_EXTENSIONS, ImageError, prepare_image
from .media import ImageData
from .pdf_files import PDF_EXTENSIONS, PDF_MAX_BYTES, PdfError, prepare_pdf
from .text_files import TEXT_EXTENSIONS, TEXT_FILE_MAX_BYTES, TextFileError, decode_text_file

logger = logging.getLogger("chatbot.discord")  # même nom qu'avant : le journal ne change pas

# Images lues par message : chacune coûte jusqu'à ~1 600 tokens.
MAX_IMAGES = 4
# Images décodées à la fois, tous messages confondus : chacune peut occuper jusqu'à 400 Mo
# le temps d'être réduite, pour une fraction de seconde.
MAX_IMAGES_AT_ONCE = 1
# Délai d'un téléchargement depuis Discord, en secondes (sinon c'est celui d'aiohttp : 5 minutes).
DOWNLOAD_TIMEOUT = 30
# Fichiers gardés en mémoire pour les messages en attente, tous salons confondus. Sans limite, des dizaines
# de PDF reçus pendant une absence feraient tomber le bot, et de nouveau à chaque reprise au redémarrage.
# Le conteneur de production a 512 Mo pour tout le bot, préparation des images comprise.
HELD_FILES_MAX_BYTES = 100_000_000
FILES_QUEUE_FULL = "J'ai déjà trop de fichiers en attente : renvoie ton message un peu plus tard, s'il te plaît."


class HeldFiles:
    """Place occupée par les fichiers des messages en attente. Réservée d'après la taille annoncée par Discord,
    avant le téléchargement : les messages reçus en même temps ne peuvent pas dépasser ensemble le budget."""

    def __init__(self, limit: int = HELD_FILES_MAX_BYTES):
        self.limit = limit
        self._sizes: dict[int, int] = {}  # message : octets

    @property
    def total(self) -> int:
        return sum(self._sizes.values())

    def reserve(self, message_id: int, size: int) -> bool:
        """Réserve la place d'un message, ou renvoie False s'il n'y en a plus. Un message sans fichier passe toujours."""
        if size and self.total + size > self.limit:
            return False
        self._sizes[message_id] = size
        return True

    def settle(self, message_id: int, size: int) -> None:
        """Message téléchargé : compte sa taille réelle (une photo réduite pèse moins que l'original).
        Sans effet si sa place a déjà été libérée (message supprimé pendant le téléchargement)."""
        if message_id in self._sizes:
            self._sizes[message_id] = size

    def release(self, *message_ids: int) -> None:
        """Libère la place de ces messages (sans effet pour ceux qui n'en ont plus)."""
        for message_id in message_ids:
            self._sizes.pop(message_id, None)


async def fetch(attachment: discord.Attachment, kind: str, error: type[UserFacingError]) -> bytes:
    """Télécharge une pièce jointe (`kind` : « image », « PDF », « fichier »). En cas d'échec
    (erreur HTTP, coupure réseau, délai dépassé), lève `error` qui demande de réessayer."""
    try:
        return await asyncio.wait_for(attachment.read(), DOWNLOAD_TIMEOUT)
    except (discord.HTTPException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.error("Téléchargement impossible (%s « %s ») : %r", kind, attachment.filename, exc)
        raise error(f"Je n'ai pas réussi à télécharger ton {kind}, réessaie.")


def looks_like_image(attachment: discord.Attachment) -> bool:
    # Un .svg est annoncé comme une image (image/svg+xml), mais c'est du texte : il est lu comme tel.
    if looks_like_text(attachment):
        return False
    return (attachment.content_type or "").startswith("image/") or attachment.filename.lower().endswith(IMAGE_EXTENSIONS)


def looks_like_text(attachment: discord.Attachment) -> bool:
    return attachment.filename.lower().endswith(TEXT_EXTENSIONS)


def looks_like_pdf(attachment: discord.Attachment) -> bool:
    return attachment.content_type == "application/pdf" or attachment.filename.lower().endswith(PDF_EXTENSIONS)


def looks_like_video(attachment: discord.Attachment) -> bool:
    return (attachment.content_type or "").startswith("video/")


class Picked(NamedTuple):
    """Pièces jointes d'un message triées : ce que le bot lit, et ce qu'il ignore (voir pick)."""
    images: list[discord.Attachment]
    pdf: discord.Attachment | None
    texts: list[discord.Attachment]
    unread: list[str]  # repères pour Claude : ce qu'il n'a pas vu, pour qu'il puisse le dire lui-même


def unread_marker(attachment: discord.Attachment) -> str:
    """Repère d'une pièce jointe que le bot ne sait pas lire (vidéo, archive…)."""
    if looks_like_video(attachment):
        return f"[vidéo jointe « {attachment.filename} » : tu ne peux pas voir les vidéos]"
    return f"[fichier joint « {attachment.filename} » : tu ne peux pas ouvrir ce type de fichier]"


def pick(attachments: list[discord.Attachment], read_pdfs: bool) -> Picked:
    """Choisit ce qui est lu, la même chose dans le salon dédié et dans les mentions : les MAX_IMAGES premières
    images, à défaut le premier PDF (read_pdfs : PDF acceptés par le .env), et les fichiers texte. Le reste
    n'empêche pas de répondre au message : il est signalé à Claude."""
    images = [a for a in attachments if looks_like_image(a)]
    pdfs = [a for a in attachments if looks_like_pdf(a)] if read_pdfs else []
    texts = [a for a in attachments if looks_like_text(a)]
    unread = [unread_marker(a) for a in attachments if a not in images + pdfs + texts]
    if len(images) > MAX_IMAGES:
        extra = len(images) - MAX_IMAGES
        unread.append(f"[{extra} autre{'s' if extra > 1 else ''} image{'s' if extra > 1 else ''} "
                      f"jointe{'s' if extra > 1 else ''}, que tu n'as pas vue{'s' if extra > 1 else ''} : "
                      f"{MAX_IMAGES} au maximum par message]")
    pdf = None if images else (pdfs[0] if pdfs else None)
    for skipped in pdfs[1:] if pdf else pdfs:
        unread.append(f"[PDF joint « {skipped.filename} », que tu n'as pas lu : un seul PDF par message, "
                      "et seulement sans image]")
    return Picked(images[:MAX_IMAGES], pdf, texts, unread)


def check_text_size(attachments: list[discord.Attachment]) -> None:
    """Lève TextFileError si ces fichiers texte dépassent ensemble TEXT_FILE_MAX_BYTES : vérifié sur la taille
    annoncée par Discord, avant tout téléchargement."""
    if sum(a.size for a in attachments) <= TEXT_FILE_MAX_BYTES:
        return
    limit = TEXT_FILE_MAX_BYTES // 1000
    if len(attachments) == 1:
        raise TextFileError(f"Ce fichier texte est trop long (maximum {limit} Ko).")
    raise TextFileError(f"Ces fichiers texte sont trop longs ensemble (maximum {limit} Ko au total).")


def text_budget(attachments: list[discord.Attachment]) -> int:
    """Place, en octets, que les fichiers texte d'un message laissent à ceux du message qu'il cite."""
    return max(0, TEXT_FILE_MAX_BYTES - sum(a.size for a in attachments if looks_like_text(a)))


def readable_size(attachments: list[discord.Attachment]) -> int:
    """Taille annoncée des fichiers que le bot peut lire : les autres (vidéos…) ne sont jamais téléchargés."""
    return sum(a.size for a in attachments if looks_like_image(a) or looks_like_pdf(a) or looks_like_text(a))


def text_file_prompt(name: str, content: str) -> str:
    """Contenu d'un fichier .txt tel qu'il est transmis à Claude."""
    return f"\n\n[Contenu du fichier joint « {name} » :]\n{content}\n[Fin du fichier « {name} »]"


def quoted_text_prompt(name: str, content: str) -> str:
    """Contenu d'un fichier texte du message cité, tel qu'il est transmis à Claude."""
    return f"\n\n[Contenu du fichier « {name} » du message cité :]\n{content}\n[Fin du fichier « {name} »]"


def text_file_marker(name: str) -> str:
    """Remplace un fichier texte dans la mémoire (son contenu n'est pas conservé)."""
    return f"[fichier texte joint « {name} » : tu l'as lu à ce moment-là, mais son contenu n'est plus conservé en mémoire]"


def attachments_marker(attachments: list[ImageData]) -> str:
    """Remplace les images ou le PDF d'un message dans la mémoire (leur contenu n'est pas conservé)."""
    if attachments[0].is_pdf:
        return f"[PDF joint « {attachments[0].name} » : tu l'as lu à ce moment-là, mais son contenu n'est plus conservé en mémoire]"
    return images_marker(len(attachments))


class AttachmentReader:
    """Télécharge et prépare les pièces jointes, selon les limites du .env (taille des images, pages des PDF)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._image_slots = asyncio.Semaphore(MAX_IMAGES_AT_ONCE)

    async def read(
        self, attachments: list[discord.Attachment]
    ) -> tuple[list[ImageData], list[tuple[str, str]], list[str]]:
        """Télécharge les pièces jointes que le bot lit (voir pick) : images ou PDF, et fichiers texte.

        Renvoie (images ou PDF, éventuellement aucun ; (nom, contenu) de chaque fichier texte ; repères de ce qui
        n'est pas lu). Lève ImageError, PdfError ou TextFileError si un fichier lu pose problème (trop lourd…).
        """
        picked = pick(attachments, read_pdfs=bool(self.settings.pdf_max_pages))
        check_text_size(picked.texts)
        if picked.images:
            files = await self.download_images(picked.images)
        else:
            files = [await self.download_pdf(picked.pdf)] if picked.pdf else []
        texts = await self.download_text_files(picked.texts) if picked.texts else []
        return files, texts, picked.unread

    async def download_text_files(self, attachments: list[discord.Attachment]) -> list[tuple[str, str]]:
        """Télécharge des fichiers texte en parallèle et renvoie (nom, contenu) de chacun, dans l'ordre du message
        (lève TextFileError s'ils sont trop longs ensemble ou si l'un ne se télécharge pas)."""
        check_text_size(attachments)
        results = await asyncio.gather(*(self.download_text_file(a) for a in attachments), return_exceptions=True)
        for result in results:  # la première erreur dans l'ordre du message, pas la plus rapide
            if isinstance(result, BaseException):
                raise result
        return list(results)

    async def download_text_file(self, attachment: discord.Attachment) -> tuple[str, str]:
        """Télécharge un fichier texte et renvoie (nom, contenu) (lève TextFileError en cas de problème)."""
        data = await fetch(attachment, "fichier", TextFileError)
        return attachment.filename, decode_text_file(data)

    async def download_pdf(self, attachment: discord.Attachment) -> ImageData:
        """Télécharge et vérifie un PDF (lève PdfError en cas de problème)."""
        if attachment.size > PDF_MAX_BYTES:
            raise PdfError(f"Ce PDF est trop lourd (maximum {PDF_MAX_BYTES // 1_000_000} Mo).")

        data = await fetch(attachment, "PDF", PdfError)
        # Lecture du PDF dans un thread : ne bloque pas le bot.
        return await asyncio.to_thread(prepare_pdf, attachment.filename, data, self.settings.pdf_max_pages)

    async def download_image(self, attachment: discord.Attachment) -> ImageData:
        """Télécharge et prépare une image (lève ImageError en cas de problème)."""
        if attachment.size > self.settings.max_image_size:
            max_mb = self.settings.max_image_size / 1_000_000
            raise ImageError(f"Cette image est trop lourde (maximum {max_mb:.2f} Mo).")

        data = await fetch(attachment, "image", ImageError)
        return await self._prepare(prepare_image, data)

    async def _prepare(self, prepare, data: bytes):
        """Réduit / convertit une image si besoin (dans un thread : ne bloque pas le bot), une à la fois
        pour borner la mémoire. Les téléchargements, eux, restent en parallèle."""
        await self._image_slots.acquire()
        preparing = asyncio.ensure_future(asyncio.to_thread(prepare, data))
        # Place rendue à la fin du thread, pas de l'attente : si l'attente est annulée,
        # le thread, lui, continue de décoder.
        preparing.add_done_callback(self._release_image_slot)
        return await asyncio.shield(preparing)

    def _release_image_slot(self, preparing: asyncio.Future) -> None:
        self._image_slots.release()
        if not preparing.cancelled():
            preparing.exception()  # lue : après une annulation, personne d'autre ne la lira

    async def download_images(self, attachments: list[discord.Attachment]) -> list[ImageData]:
        """Télécharge plusieurs images en parallèle et les prépare (lève ImageError si l'une pose problème).

        Si plusieurs échouent, l'erreur nomme la première dans l'ordre du message, pas la plus rapide.
        """
        results = await asyncio.gather(*(self.download_image(a) for a in attachments), return_exceptions=True)
        for attachment, result in zip(attachments, results):
            if isinstance(result, ImageError) and len(attachments) > 1:
                # précise laquelle, sinon « cette image » est ambigu
                raise ImageError(f"« {attachment.filename} » : {result.user_message}")
            if isinstance(result, BaseException):
                raise result
        return list(results)

    async def quoted_images(self, attachments: list[discord.Attachment], kind: str) -> list[ImageData]:
        """Télécharge les MAX_IMAGES premières images d'un message cité ou transféré (les illisibles sont ignorées)."""
        images = [a for a in attachments if looks_like_image(a)][:MAX_IMAGES]
        results = await asyncio.gather(*(self.download_image(a) for a in images), return_exceptions=True)
        for result in results:
            if isinstance(result, ImageError):
                logger.warning("Image du message %s ignorée : %s", kind, result.user_message)
            elif isinstance(result, BaseException):
                raise result
        return [r for r in results if isinstance(r, ImageData)]

    async def quoted_text_files(self, attachments: list[discord.Attachment],
                                budget: int) -> tuple[list[tuple[str, str]], list[str]]:
        """Fichiers texte d'un message cité, tant qu'ils tiennent dans budget octets (ce que les fichiers du
        message lui-même laissent des TEXT_FILE_MAX_BYTES). Renvoie (nom, contenu) de ceux qui sont lus, et le
        nom de ceux qui ne tiennent pas. Un fichier qui ne se télécharge pas est ignoré."""
        kept, too_long = [], []
        for attachment in (a for a in attachments if looks_like_text(a)):
            if attachment.size <= budget:
                kept.append(attachment)
                budget -= attachment.size
            else:
                too_long.append(attachment.filename)
        results = await asyncio.gather(*(self.download_text_file(a) for a in kept), return_exceptions=True)
        read = []
        for result in results:
            if isinstance(result, TextFileError):
                logger.warning("Fichier texte du message cité ignoré : %s", result.user_message)
            elif isinstance(result, BaseException):
                raise result
            else:
                read.append(result)
        return read, too_long

    async def read_gif(self, embeds: list[discord.Embed]) -> tuple[ImageData, str, str] | None:
        """GIF choisi dans le sélecteur de Discord (voir gifs.py) : (image, explication pour Claude, repère pour
        la mémoire), ou None s'il n'y en a pas ou s'il est illisible (le message est alors lu sans lui)."""
        embed = gif_embed(embeds)
        url = gif_source(embed) if embed else None
        if url is None:
            return None
        data = await download_gif(url)
        if data is None:
            return None
        try:
            image, frames = await self._prepare(prepare_gif, data)
        except ImageError as exc:
            logger.warning("GIF ignoré (%s) : %s", url, exc.user_message)
            return None
        title = gif_title(embed)
        picker = from_picker(embed)
        marker = gif_marker(title) if picker or frames > 1 else images_marker(1)  # photo partagée par lien
        return image, gif_prompt(title, frames, picker), marker
