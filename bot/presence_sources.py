"""Animes, musiques et jeux réels pour le statut Discord, lus sur AniList, Deezer et Steam."""

from __future__ import annotations

import asyncio
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
STEAM_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
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
                 steam_api_key: str | None, steam_id: str | None):
        self.cache_file = cache_file
        self.deezer_playlists = deezer_playlists
        self.steam_api_key = steam_api_key
        self.steam_id = steam_id
        self.anime: list[Anime] = []
        self.music: list[Track] = []
        self.games: list[str] = []

    @property
    def steam_enabled(self) -> bool:
        return bool(self.steam_api_key and self.steam_id)

    def load_cache(self) -> None:
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            self.anime = [Anime(**a) for a in data.get("anime", [])]
            self.music = [Track(**t) for t in data.get("music", [])]
            if self.steam_enabled:
                self.games = list(data.get("games", []))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Cache des statuts %s illisible, ignoré : %s", self.cache_file, exc)

    def save_cache(self) -> None:
        data = {"anime": [asdict(a) for a in self.anime], "music": [asdict(t) for t in self.music], "games": self.games}
        try:
            self.cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
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

    async def fetch_games(self, session: aiohttp.ClientSession) -> list[str]:
        params = {"key": self.steam_api_key, "steamid": self.steam_id, "include_appinfo": "1",
                  "include_played_free_games": "1", "format": "json"}
        data = await get_json(session, "GET", STEAM_URL, params=params)
        games = data.get("response", {}).get("games")
        if not games:
            raise SourceError("bibliothèque vide ou privée : « Détails des jeux » doit être public dans le profil Steam")
        return sorted({g["name"] for g in games if g.get("name")})
