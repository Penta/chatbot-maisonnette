"""GIF choisis dans le sélecteur de Discord (Klipy, Giphy…, ou les favoris du membre).

Ils ne sont pas joints au message : Discord envoie un simple lien, puis l'affiche avec un aperçu (« embed »
de type gifv, ou image pour un favori, dont le lien mène à une copie gardée par Discord). Claude ne lit pas les vidéos : d'un GIF animé, il reçoit quelques images prises du début à la
fin de l'animation, assemblées en une grille, ce qui lui permet d'en suivre le déroulé.
"""

from __future__ import annotations

import asyncio
import io
import logging
from urllib.parse import urlsplit

import aiohttp
import discord
from PIL import Image

from .images import MAX_PIXELS, ImageError, prepare_image
from .media import ImageData

logger = logging.getLogger("chatbot.discord")

GIF_FRAMES = 6  # images prises dans l'animation
GRID_COLUMNS = 3
CELL_SIZE = 360  # grand côté de chaque image de la grille : 1 080 × 720 au plus, environ 1 000 tokens
GRID_GAP = 6  # trait blanc entre les images de la grille
GIF_MAX_BYTES = 10_000_000
# Au-delà, l'animation n'est pas parcourue (décoder des milliers d'images bloquerait le bot) : première image.
GIF_MAX_TOTAL_PIXELS = 300_000_000
DOWNLOAD_TIMEOUT = 30
# Sites dont l'aperçu est téléchargé chez eux, animé. La copie que Discord en garde (proxy_url) n'a plus que
# la première image, mais elle seule est prise ailleurs : le bot ne télécharge rien depuis un site inconnu.
ANIMATED_HOSTS = {"static.klipy.com"}
DISCORD_PROXY_SUFFIX = ".discordapp.net"
# Serveurs de Discord (media.discordapp.net, images-ext-1.discordapp.net, cdn.discordapp.com) : un GIF des
# favoris est un lien vers l'un d'eux, téléchargé tel quel (il garde l'animation, contrairement à proxy_url).
DISCORD_SITES = ("discordapp.net", "discordapp.com")
# Liens du sélecteur de GIF de Discord. Leur aperçu peut arriver juste après le message (voir
# ChatBot.read_gif) : il est attendu jusqu'à EMBED_WAIT secondes.
GIF_SITES = ("klipy.com", "giphy.com", "tenor.com") + DISCORD_SITES
EMBED_WAIT = 3.0


def _host(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    return parts.hostname if parts.scheme == "https" else None


def _on(url: str | None, sites: tuple[str, ...]) -> bool:
    """L'adresse est en https sur l'un de ces sites (ou l'un de leurs sous-domaines)."""
    host = _host(url)
    return bool(host) and any(host == site or host.endswith("." + site) for site in sites)


def gif_embed(embeds: list[discord.Embed]) -> discord.Embed | None:
    """Aperçu d'un GIF du sélecteur de Discord parmi ceux du message, ou None. Une image n'en est un que si
    elle est gardée par Discord (GIF des favoris) : celle d'un autre site n'est pas lue."""
    return next((embed for embed in embeds
                 if embed.type == "gifv" or (embed.type == "image" and _on(embed.url, DISCORD_SITES))), None)


def looks_like_gif_link(content: str) -> bool:
    """Message fait d'un seul lien vers un site de GIF (celui qu'envoie le sélecteur de Discord)."""
    words = content.split()
    return len(words) == 1 and _on(words[0], GIF_SITES)


def gif_source(embed: discord.Embed) -> str | None:
    """Adresse de l'image du GIF à télécharger : animée chez Klipy ou chez Discord (favoris), sinon la copie
    fixe gardée par Discord."""
    if _host(embed.thumbnail.url) in ANIMATED_HOSTS or _on(embed.thumbnail.url, DISCORD_SITES):
        return embed.thumbnail.url
    proxy = embed.thumbnail.proxy_url
    host = _host(proxy)
    return proxy if host and host.endswith(DISCORD_PROXY_SUFFIX) else None


def gif_title(embed: discord.Embed) -> str | None:
    return (embed.title or "").strip() or None


def gif_prompt(title: str | None, frames: int) -> str:
    """Explication du GIF pour Claude, jointe au message (pas conservée en mémoire)."""
    name = f"GIF « {title} »" if title else "GIF"
    if frames > 1:
        return (f"\n[{name} envoyé avec le sélecteur de Discord : l'image jointe montre {frames} moments de "
                "l'animation, du début à la fin, à lire de gauche à droite puis de haut en bas]")
    return f"\n[{name} envoyé avec le sélecteur de Discord : seule sa première image est jointe]"


def gif_marker(title: str | None) -> str:
    """Remplace le GIF dans la mémoire (son image n'est pas conservée)."""
    name = f"GIF « {title} »" if title else "GIF"
    return f"[{name} : tu l'as vu à ce moment-là, mais il n'est plus conservé en mémoire]"


async def download_gif(url: str) -> bytes | None:
    """Télécharge l'image d'un GIF, ou renvoie None en cas d'échec (le message est lu sans elle)."""
    try:
        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Sans redirection : le téléchargement reste sur le site vérifié par gif_source.
            async with session.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    logger.warning("GIF non téléchargé (%s) : %s", response.status, url)
                    return None
                if (response.content_length or 0) > GIF_MAX_BYTES:
                    logger.warning("GIF trop lourd (%s octets) : %s", response.content_length, url)
                    return None
                # Lu morceau par morceau jusqu'au bout (un seul read() s'arrête au premier morceau reçu),
                # en s'arrêtant dès que la limite est dépassée.
                data = bytearray()
                async for chunk in response.content.iter_chunked(64_000):
                    data += chunk
                    if len(data) > GIF_MAX_BYTES:
                        logger.warning("GIF trop lourd : %s", url)
                        return None
                return bytes(data)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("GIF non téléchargé : %s (%r)", url, exc)
        return None


def prepare_gif(data: bytes) -> tuple[ImageData, int]:
    """Prépare l'image d'un GIF pour Claude : une grille de GIF_FRAMES moments de l'animation, ou l'image
    seule si elle n'est pas animée. Renvoie (image, nombre de moments montrés). Lève ImageError.

    Fonction bloquante (calculs sur l'image) : à appeler via asyncio.to_thread.
    """
    try:
        return _prepare_gif(data)
    except ImageError:
        raise
    except Exception as exc:
        # Sur un fichier coupé ou abîmé, Pillow lève des erreurs de toutes sortes (IndexError en comptant les
        # images d'un GIF coupé…) : le GIF est illisible, le message est lu sans lui.
        raise ImageError("GIF illisible.") from exc


def _prepare_gif(data: bytes) -> tuple[ImageData, int]:
    img = Image.open(io.BytesIO(data))
    count = getattr(img, "n_frames", 1)
    pixels = img.width * img.height
    if count < 2 or pixels > MAX_PIXELS or pixels * count > GIF_MAX_TOTAL_PIXELS:
        return prepare_image(data), 1

    shown = min(GIF_FRAMES, count)
    indexes = [round(i * (count - 1) / (shown - 1)) for i in range(shown)]
    frames = []
    for index in indexes:
        img.seek(index)
        frame = img.convert("RGBA")
        background = Image.new("RGB", frame.size, "white")  # transparence : fond blanc
        background.paste(frame, mask=frame.getchannel("A"))
        background.thumbnail((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)
        frames.append(background)

    columns = min(GRID_COLUMNS, shown)
    rows = -(-shown // columns)
    width, height = frames[0].size
    grid = Image.new("RGB", (columns * width + (columns - 1) * GRID_GAP, rows * height + (rows - 1) * GRID_GAP),
                     "white")
    for position, frame in enumerate(frames):
        row, column = divmod(position, columns)
        grid.paste(frame, (column * (width + GRID_GAP), row * (height + GRID_GAP)))
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=85, optimize=True)
    return ImageData("image/jpeg", buffer.getvalue()), shown
