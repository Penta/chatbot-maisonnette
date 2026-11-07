"""Particularités de chaque modèle Claude : réglage de la réflexion et tarifs."""

from __future__ import annotations

import re

# « claude-sonnet-5-5 » ou « claude-sonnet-4-5-20250929 » : famille, version, date éventuelle.
_MODEL_ID = re.compile(r"^claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$")

# Façon de couper la réflexion préalable (THINKING=off), qui dépend du modèle.
# None : impossible sur ce modèle, seul EFFORT permet alors de la réduire.
_THINKING_OFF: dict[str, dict | None] = {
    "claude-sonnet-5-5": {"type": "between_tools"},  # n'existe que sur Sonnet 5.5
    "claude-opus-5-5": None,
}
# Haiku 5.5 et les modèles plus anciens (Sonnet 5, Opus 4.8…).
_DEFAULT_THINKING_OFF = {"type": "disabled"}

# Tarifs en dollars par million de tokens (entrée, sortie), pour l'estimation des coûts.
# Haiku 5.5 : tarif des requêtes de moins de 100 000 tokens (0,50 / 2,50 au-delà).
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.0, 20.0),
}

# Prix de la lecture depuis le cache, en part du prix d'entrée, quand il diffère des 10 % habituels.
_CACHE_READ_RATIOS: dict[str, float] = {
    "claude-opus-5-5": 0.05,  # 0,20 $ par million de tokens
}
_DEFAULT_CACHE_READ_RATIO = 0.1


def is_known(model: str) -> bool:
    """Modèle connu du bot (tarifs et réglage de la réflexion) : sinon, peut-être une faute de frappe."""
    return model in PRICES


def display_name(model: str) -> str:
    """Nom lisible du modèle (« Sonnet 5.5 ») ; l'identifiant tel quel s'il n'a pas la forme habituelle."""
    if not (match := _MODEL_ID.match(model)):
        return model
    family, major, minor = match.groups()
    return f"{family.capitalize()} {major}{f'.{minor}' if minor else ''}"


def thinking_off(model: str) -> dict | None:
    """Paramètre « thinking » à envoyer quand THINKING=off (None : ne rien envoyer)."""
    return _THINKING_OFF.get(model, _DEFAULT_THINKING_OFF)


def cache_read_ratio(model: str) -> float:
    """Part du prix d'entrée payée pour un token relu depuis le cache."""
    return _CACHE_READ_RATIOS.get(model, _DEFAULT_CACHE_READ_RATIO)
