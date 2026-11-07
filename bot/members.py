"""Registre des membres vus par le bot (noms, anciens pseudos, réglages de la mémoire), dans un fichier JSON."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from .files import write_atomic

logger = logging.getLogger("chatbot.members")

# Version 2 : mémoire à activer soi-même avec /memoire (depuis la 5.11).
FILE_VERSION = 2
# Noms Discord d'un membre : nom d'utilisateur (unique), nom d'affichage, pseudo sur le serveur.
NAME_KEYS = ("username", "global_name", "nickname")
# Anciens pseudos gardés par membre, du plus ancien au plus récent.
MAX_FORMER_NAMES = 5


class MemberRegistry:
    """Garde, pour chaque membre vu (auteur d'un message lu par le bot, ou membre mentionné), ses noms
    actuels et ses anciens pseudos, et s'il a activé la mémoire (désactivée par défaut).

    Le fichier n'est réécrit que si quelque chose change : pas d'écriture à chaque message."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        # Raison pour laquelle le fichier est illisible, ou None. Repartir d'un registre vide oublierait
        # qui a désactivé la mémoire : la mémoire est alors coupée, et le fichier n'est jamais réécrit.
        self.broken: str | None = None
        self._data: dict[str, dict] = self._load()
        self._save_lock = asyncio.Lock()
        # Dernière sauvegarde ratée (disque plein, droits) : le fichier est en retard sur _data. La
        # suivante, quelle qu'elle soit, réécrit tout et rattrape ce retard.
        self._unsaved = False

    # ------------------------------------------------------------------ accès

    def get(self, member_id: int | str) -> dict | None:
        return self._data.get(str(member_id))

    def members(self) -> dict[str, dict]:
        """Tous les membres connus, par identifiant."""
        return self._data

    def memory_enabled(self, member_id: int | str) -> bool:
        """Le membre a activé la mémoire avec /memoire : sans cela, rien n'est noté sur lui."""
        entry = self.get(member_id)
        return entry is not None and entry["memory"]

    async def see(self, member_id: int, username: str, global_name: str | None,
                  nickname: str | None = None, in_guild: bool = True) -> None:
        """Enregistre un membre ou met ses noms à jour ; un nom remplacé rejoint les anciens pseudos.

        in_guild=False (message privé) : le pseudo sur le serveur n'est pas connu, il reste inchangé."""
        names = {"username": username, "global_name": global_name or None}
        if in_guild:
            names["nickname"] = nickname or None
        key = str(member_id)
        entry = self._data.get(key)
        if entry is None:
            self._data[key] = {
                **{name_key: None for name_key in NAME_KEYS}, **names,
                "former_names": [], "memory": False,
                "first_seen": datetime.now().strftime("%Y-%m-%d"),
            }
            logger.info("Nouveau membre dans le registre : %s (%s)", display_name(self._data[key]), key)
            await self._save()
            return
        old_names = [entry[name_key] for name_key in NAME_KEYS if entry[name_key]]
        changed = False
        for name_key, value in names.items():
            if entry[name_key] != value:
                entry[name_key] = value
                changed = True
        if not changed:
            return
        current = {entry[name_key] for name_key in NAME_KEYS if entry[name_key]}
        former = [name for name in entry["former_names"] if name not in current]
        for name in old_names:
            if name not in current and name not in former:
                former.append(name)
        entry["former_names"] = former[-MAX_FORMER_NAMES:]
        logger.info("Noms mis à jour dans le registre : %s (%s)", display_name(entry), key)
        await self._save()

    async def set_memory(self, member_id: int | str, enabled: bool) -> bool:
        """Active ou désactive la mémoire pour ce membre. Renvoie False si le réglage n'a pas pu être
        enregistré : il s'applique quand même, mais serait perdu à un redémarrage avant la prochaine
        sauvegarde réussie."""
        if (entry := self.get(member_id)) is None:
            return True
        if entry["memory"] != enabled:
            entry["memory"] = enabled
            logger.info("Mémoire %s pour %s (%s)", "activée" if enabled else "désactivée", display_name(entry), member_id)
        elif not self._unsaved:
            return True
        return await self._save()

    # ------------------------------------------------------------ internes

    def _load(self) -> dict[str, dict]:
        if not self.file_path.exists():
            return {}
        try:
            # -sig : BOM laissé par certains éditeurs Windows (Bloc-notes) après une correction à la main.
            raw = json.loads(self.file_path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or not isinstance(raw.get("members"), dict):
                raise ValueError("« members » absent ou n'est pas un dictionnaire")
            if raw.get("version") != FILE_VERSION:
                raise ValueError(f"format {raw.get('version')!r} inconnu (attendu : {FILE_VERSION})")
            data = {key: self._sanitize(key, value) for key, value in raw["members"].items()}
        except (ValueError, OSError) as exc:  # JSON invalide, mauvais encodage ou structure inattendue
            self.broken = str(exc)
            logger.error("Registre des membres illisible (%s) : mémoire des membres coupée jusqu'à sa "
                         "réparation. Corrige %s puis redémarre le bot ; le supprimer repartirait de zéro, "
                         "en désactivant la mémoire de tous ceux qui l'avaient activée.", exc, self.file_path)
            return {}
        return data

    @staticmethod
    def _sanitize(key: str, value) -> dict:
        """Une entrée du fichier, complétée si besoin. ValueError si elle est illisible, ou si on ne sait
        pas si la mémoire est activée : mieux vaut que l'administrateur corrige le fichier."""
        if not isinstance(value, dict) or not isinstance(value.get("username"), str):
            raise ValueError(f"membre {key} : entrée illisible")
        if not isinstance(value.get("memory"), bool):
            raise ValueError(f"membre {key} : « memory » doit valoir true ou false")
        entry = {name_key: value.get(name_key) if isinstance(value.get(name_key), str) else None for name_key in NAME_KEYS}
        former = value.get("former_names")
        entry["former_names"] = [n for n in former if isinstance(n, str)] if isinstance(former, list) else []
        entry["memory"] = value["memory"]
        entry["first_seen"] = value.get("first_seen") if isinstance(value.get("first_seen"), str) else None
        return entry

    async def _save(self) -> bool:
        """Écrit le registre ; renvoie False s'il n'a pas pu l'être."""
        if self.broken:
            return False  # le fichier illisible attend sa réparation : l'écraser perdrait les désactivations
        payload = {"version": FILE_VERSION, "members": self._data}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        async with self._save_lock:
            try:
                await asyncio.to_thread(write_atomic, self.file_path, text)
            except OSError as exc:
                logger.error("Impossible de sauvegarder le registre des membres : %s", exc)
                self._unsaved = True
                return False
            self._unsaved = False
            return True


def display_name(entry: dict) -> str:
    """Le nom affiché sur le serveur : pseudo, sinon nom d'affichage, sinon nom d'utilisateur."""
    return entry["nickname"] or entry["global_name"] or entry["username"]


def with_username(name: str, username: str) -> str:
    """« Johnounet (johnk) » : le nom affiché, libre et non unique, suivi du nom d'utilisateur, unique,
    pour que personne ne puisse se faire passer pour un autre en prenant son pseudo. Seul s'il est
    identique au nom affiché (« Penta » et « penta »)."""
    return name if name.casefold() == username.casefold() else f"{name} ({username})"
