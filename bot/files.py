"""Écriture des fichiers gardés sur le disque."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


def write_atomic(path: Path, text: str, flush_to_disk: bool = True) -> None:
    """Écrit text dans path sans jamais laisser de fichier à moitié écrit : dans un fichier temporaire,
    envoyé jusqu'au disque (fsync) avant d'être renommé, sinon une coupure de courant juste après le
    renommage pourrait laisser un fichier vide. Le fichier temporaire est retiré si l'écriture échoue.
    flush_to_disk=False : sans fsync, parfois lent sur un volume réseau. Le fichier résiste toujours à
    l'arrêt brutal du programme, mais plus à une coupure de courant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as file:
            file.write(text)
            if flush_to_disk:
                file.flush()
                os.fsync(file.fileno())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
