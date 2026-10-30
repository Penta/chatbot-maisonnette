"""Animes, musiques et jeux réels pour le statut Discord, lus sur AniList, Deezer et Steam."""

from __future__ import annotations

import asyncio
import html
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp

logger = logging.getLogger("chatbot.presence")

ANILIST_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    media(type: ANIME, sort: POPULARITY_DESC, isAdult: false) {
      title { romaji } episodes duration nextAiringEpisode { episode }
    }
  }
}
"""
ANIME_PAGES = 10  # 10 pages de 50 : le top 500 des animes les plus populaires
# Les tops Deezer contiennent 100 titres.
DEEZER_TRACKS_URL = "https://api.deezer.com/playlist/{}/tracks?limit=100"
STEAM_APPS_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
STEAM_PAGE_SIZE = 50_000  # maximum accepté par Steam
STEAM_MAX_PAGES = 10
STEAM_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
# Descripteurs de contenu Steam écartés : 3 = contenu sexuel réservé aux adultes,
# 4 = nudité ou contenu sexuel fréquents (1 = « un peu de nudité » reste accepté).
STEAM_EXCLUDED_DESCRIPTORS = {3, 4}
STEAM_DESCRIPTION_MAX_LENGTH = 300
DEFAULT_TRACK_DURATION = 180  # secondes, si Deezer n'indique pas la durée
DEFAULT_EPISODE_DURATION = 24  # minutes, si AniList n'indique pas la durée
TIMEOUT = aiohttp.ClientTimeout(total=30)


class SourceError(Exception):
    """Lecture d'une source impossible. Le message ne contient jamais la clé Steam."""


@dataclass(frozen=True)
class Anime:
    title: str
    episodes: int | None  # épisodes déjà sortis (None : inconnu) ; 1 pour un film
    duration: int         # minutes par épisode


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
        # Pas de str(exc) : l'adresse de la requête, qu'il peut contenir, porte la clé Steam.
        raise SourceError(type(exc).__name__) from None


class PresenceSources:
    """Listes téléchargées au démarrage puis une fois par jour, gardées en cache sur le disque.

    Si une source ne répond pas, sa liste précédente (ou celle du cache) est conservée."""

    def __init__(self, cache_file: Path, deezer_playlists: tuple[int, ...],
                 steam_api_key: str | None):
        self.cache_file = cache_file
        self.deezer_playlists = deezer_playlists
        self.steam_api_key = steam_api_key
        self.anime: list[Anime] = []
        self.music: list[Track] = []
        self.games: list[int] = []  # identifiants de tous les jeux de la boutique Steam

    @property
    def steam_enabled(self) -> bool:
        return bool(self.steam_api_key)

    def load_cache(self) -> None:
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            self.anime = [Anime(**a) for a in data.get("anime", [])]
            self.music = [Track(**t) for t in data.get("music", [])]
            if self.steam_enabled:
                self.games = [g for g in data.get("games", []) if isinstance(g, int)]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Cache des statuts %s illisible, ignoré : %s", self.cache_file, exc)

    def save_cache(self) -> None:
        data = {"anime": [asdict(a) for a in self.anime], "music": [asdict(t) for t in self.music], "games": self.games}
        try:
            # Sans indentation : la liste des jeux compte plus de 100 000 identifiants.
            self.cache_file.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        except OSError as exc:
            logger.warning("Impossible d'écrire le cache des statuts %s : %s", self.cache_file, exc)

    async def refresh(self) -> None:
        sources = [("anime", "Animes (AniList)", self.fetch_anime), ("music", "Musiques (Deezer)", self.fetch_music)]
        if self.steam_enabled:
            sources.append(("games", "Jeux (Steam)", self.fetch_games))
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            for attr, label, fetch in sources:
                try:
                    items = await fetch(session)
                    if not items:
                        raise SourceError("liste vide")
                except SourceError as exc:
                    logger.warning("%s : lecture impossible (%s), %d élément(s) conservé(s).",
                                   label, exc, len(getattr(self, attr)))
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
                # Anime en cours de diffusion : seuls les épisodes déjà sortis comptent.
                episodes = m.get("episodes")
                if episodes is None and m.get("nextAiringEpisode"):
                    episodes = m["nextAiringEpisode"]["episode"] - 1 or None
                anime[title] = Anime(title, episodes, m.get("duration") or DEFAULT_EPISODE_DURATION)
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

    async def fetch_games(self, session: aiohttp.ClientSession) -> list[int]:
        """Identifiants de tous les jeux de la boutique (sans DLC, logiciels ni vidéos), page par page."""
        games: list[int] = []
        last_appid = 0
        for _ in range(STEAM_MAX_PAGES):
            params = {"key": self.steam_api_key, "include_games": "true", "include_dlc": "false",
                      "include_software": "false", "include_videos": "false", "include_hardware": "false",
                      "max_results": str(STEAM_PAGE_SIZE), "last_appid": str(last_appid)}
            data = (await get_json(session, "GET", STEAM_APPS_URL, params=params)).get("response", {})
            games += [app["appid"] for app in data.get("apps", []) if app.get("appid")]
            if not data.get("have_more_results") or not data.get("last_appid"):
                break
            last_appid = data["last_appid"]
        return games

    @staticmethod
    async def game_details(appid: int) -> Game | None:
        """Fiche d'un jeu sur la boutique, ou None s'il ne convient pas (introuvable, pas un jeu, pour adultes)."""
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            data = await get_json(session, "GET", STEAM_DETAILS_URL,
                                  params={"appids": str(appid), "l": "french", "cc": "fr"})
        entry = data.get(str(appid)) or {}
        info = entry.get("data") if entry.get("success") else None
        if not info or info.get("type") != "game" or not info.get("name"):
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
