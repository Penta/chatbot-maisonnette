"""Animes, musiques et jeux réels pour le statut Discord, lus sur AniList, Deezer et Steam."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp

logger = logging.getLogger("chatbot.presence")

ANILIST_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    media(type: ANIME, sort: POPULARITY_DESC, isAdult: false) {
      title { romaji } format status episodes duration nextAiringEpisode { episode }
    }
  }
}
"""
ANIME_PAGES = 10  # 10 pages de 50 : le top 500 des animes les plus populaires
# Les tops Deezer contiennent 100 titres.
DEEZER_TRACKS_URL = "https://api.deezer.com/playlist/{}/tracks?limit=100"
# Recherche de la boutique Steam (celle du site, sans clé) : meilleures ventes en France, jeux seulement
# (category1=998, sans DLC ni logiciels). Elle donne le nom et l'image de chaque jeu, pas son identifiant :
# il est lu dans l'adresse de l'image (…/apps/<identifiant>/…).
STEAM_SEARCH_URL = "https://store.steampowered.com/search/results/"
STEAM_TOP_GAMES = 500  # assez pour varier, assez connus pour que les membres en parlent
STEAM_PAGE_SIZE = 100
_STEAM_APPID = re.compile(r"/apps/(\d+)/")
STEAM_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
# Descripteurs de contenu Steam écartés : 3 = contenu sexuel réservé aux adultes,
# 4 = nudité ou contenu sexuel fréquents (1 = « un peu de nudité » reste accepté).
STEAM_EXCLUDED_DESCRIPTORS = {3, 4}
STEAM_DESCRIPTION_MAX_LENGTH = 300
DEFAULT_TRACK_DURATION = 180  # secondes, si Deezer n'indique pas la durée
DEFAULT_EPISODE_DURATION = 24  # minutes, si AniList n'indique pas la durée
TIMEOUT = aiohttp.ClientTimeout(total=30)


class SourceError(Exception):
    """Lecture d'une source impossible."""


@dataclass(frozen=True)
class Anime:
    title: str
    episodes: int | None  # épisodes déjà sortis (None : inconnu)
    duration: int         # minutes par épisode
    movie: bool = False   # film ou épisode unique : affiché sans numéro d'épisode


@dataclass(frozen=True)
class Game:
    name: str
    description: str | None  # présentation courte de la boutique Steam


@dataclass(frozen=True)
class Track:
    title: str
    artist: str
    duration: int  # secondes


async def get_json(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> dict:
    try:
        async with session.request(method, url, **kwargs) as response:
            if response.status in (401, 403):
                raise SourceError(f"accès refusé (HTTP {response.status})")
            if response.status != 200:
                raise SourceError(f"réponse HTTP {response.status}")
            return await response.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        raise SourceError(type(exc).__name__) from None


class PresenceSources:
    """Listes téléchargées au démarrage puis une fois par jour, gardées en cache sur le disque.

    Si une source ne répond pas, sa liste précédente (ou celle du cache) est conservée."""

    def __init__(self, cache_file: Path, deezer_playlists: tuple[int, ...]):
        self.cache_file = cache_file
        self.deezer_playlists = deezer_playlists
        self.anime: list[Anime] = []
        self.music: list[Track] = []
        self.games: list[int] = []  # identifiants des meilleures ventes de Steam en France

    def load_cache(self) -> None:
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            # Ancien cache, sans « movie » : un seul épisode y voulait dire « film ».
            self.anime = [Anime(**{"movie": a.get("episodes") == 1, **a}) for a in data.get("anime", [])]
            self.music = [Track(**t) for t in data.get("music", [])]
            # « top_games » : l'ancienne liste « games », toute la boutique Steam, est ignorée.
            self.games = [g for g in data.get("top_games", []) if isinstance(g, int)]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError) as exc:  # AttributeError : entrée qui n'est pas une fiche
            logger.warning("Cache des statuts %s illisible, ignoré : %s", self.cache_file, exc)

    def save_cache(self) -> None:
        data = {"anime": [asdict(a) for a in self.anime], "music": [asdict(t) for t in self.music],
                "top_games": self.games}
        try:
            # Sans indentation : le fichier n'est pas fait pour être lu.
            self.cache_file.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        except OSError as exc:
            logger.warning("Impossible d'écrire le cache des statuts %s : %s", self.cache_file, exc)

    async def refresh(self) -> None:
        sources = [("anime", "Animes (AniList)", self.fetch_anime), ("music", "Musiques (Deezer)", self.fetch_music),
                   ("games", "Jeux (Steam)", self.fetch_games)]
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            for attr, label, fetch in sources:
                try:
                    items = await fetch(session)
                    if not items:
                        raise SourceError("liste vide")
                except SourceError as exc:
                    logger.warning("%s : lecture impossible (%s), %d élément(s) conservé(s).",
                                   label, exc, len(getattr(self, attr)))
                except Exception:
                    # Réponse d'une forme inattendue : sans ce filet, les autres sources seraient perdues, et
                    # au démarrage (PresenceManager._run) toute la gestion du statut s'arrêterait.
                    logger.exception("%s : réponse inattendue, %d élément(s) conservé(s).",
                                     label, len(getattr(self, attr)))
                else:
                    setattr(self, attr, items)
                    logger.info("%s : %d élément(s).", label, len(items))
        self.save_cache()

    @staticmethod
    async def fetch_anime(session: aiohttp.ClientSession) -> list[Anime]:
        anime: dict[str, Anime] = {}
        for page in range(1, ANIME_PAGES + 1):
            data = await get_json(session, "POST", ANILIST_URL,
                                  json={"query": ANILIST_QUERY, "variables": {"page": page}})
            try:
                media = data["data"]["Page"]["media"]
            except (KeyError, TypeError):
                raise SourceError("réponse inattendue") from None
            for m in media:
                title = (m.get("title") or {}).get("romaji")
                if not title:
                    continue
                status = m.get("status")
                if status == "NOT_YET_RELEASED":
                    continue
                episodes = m.get("episodes")  # total prévu, pas forcément tous sortis
                if status in ("RELEASING", "HIATUS"):
                    # En cours de diffusion : seuls les épisodes déjà sortis comptent. Sans date du
                    # prochain épisode (pause), on ne sait pas lesquels sont sortis : anime écarté.
                    next_episode = (m.get("nextAiringEpisode") or {}).get("episode")
                    if not next_episode or next_episode <= 1:
                        continue
                    episodes = next_episode - 1
                # Film, ou épisode spécial unique déjà terminé : affiché sans numéro d'épisode.
                # Une série qui n'a encore qu'un épisode sorti garde le sien.
                movie = m.get("format") == "MOVIE" or (status == "FINISHED" and episodes == 1)
                anime[title] = Anime(title, episodes, m.get("duration") or DEFAULT_EPISODE_DURATION, movie)
        return list(anime.values())

    async def fetch_music(self, session: aiohttp.ClientSession) -> list[Track]:
        tracks: dict[int, Track] = {}  # par identifiant Deezer : un titre présent dans deux tops ne compte qu'une fois
        for playlist in self.deezer_playlists:
            data = await get_json(session, "GET", DEEZER_TRACKS_URL.format(playlist))
            if "error" in data:  # Deezer signale ses erreurs avec un code HTTP 200
                raise SourceError(f"playlist {playlist} : {data['error'].get('message', 'erreur')}")
            for t in data.get("data", []):
                if t.get("id") and t.get("title") and t.get("artist", {}).get("name"):
                    tracks[t["id"]] = Track(t["title"], t["artist"]["name"], t.get("duration") or DEFAULT_TRACK_DURATION)
        return list(tracks.values())

    @staticmethod
    async def fetch_games(session: aiohttp.ClientSession) -> list[int]:
        """Identifiants des STEAM_TOP_GAMES meilleures ventes de Steam en France (jeux seulement), page par page."""
        games: dict[int, None] = {}  # dans l'ordre, sans doublon (le classement peut bouger entre deux pages)
        for start in range(0, STEAM_TOP_GAMES, STEAM_PAGE_SIZE):
            params = {"filter": "topsellers", "category1": "998", "cc": "fr", "json": "1",
                      "start": str(start), "count": str(STEAM_PAGE_SIZE)}
            data = await get_json(session, "GET", STEAM_SEARCH_URL, params=params)
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise SourceError("réponse inattendue")
            for item in items:
                logo = item.get("logo") if isinstance(item, dict) else None
                if isinstance(logo, str) and (match := _STEAM_APPID.search(logo)):
                    games[int(match.group(1))] = None
            if len(items) < STEAM_PAGE_SIZE:
                break  # fin du classement
        return list(games)[:STEAM_TOP_GAMES]

    @staticmethod
    async def game_details(appid: int) -> Game | None:
        """Fiche d'un jeu sur la boutique, ou None s'il ne convient pas (introuvable, pas un jeu, pas sorti, pour adultes)."""
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            data = await get_json(session, "GET", STEAM_DETAILS_URL,
                                  params={"appids": str(appid), "l": "french", "cc": "fr"})
        entry = data.get(str(appid)) or {}
        info = entry.get("data") if entry.get("success") else None
        if not info or info.get("type") != "game" or not info.get("name"):
            return None
        if (info.get("release_date") or {}).get("coming_soon"):
            return None
        descriptors = set((info.get("content_descriptors") or {}).get("ids") or [])
        try:
            adult = int(info.get("required_age") or 0) >= 18
        except ValueError:
            adult = False
        if adult or descriptors & STEAM_EXCLUDED_DESCRIPTORS:
            return None
        description = html.unescape(info.get("short_description") or "").strip()
        return Game(info["name"], description[:STEAM_DESCRIPTION_MAX_LENGTH] or None)
