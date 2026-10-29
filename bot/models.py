"""Particularités de chaque modèle Claude : réglage de la réflexion et tarifs."""

from __future__ import annotations

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


def thinking_off(model: str) -> dict | None:
    """Paramètre « thinking » à envoyer quand THINKING=off (None : ne rien envoyer)."""
    return _THINKING_OFF.get(model, _DEFAULT_THINKING_OFF)
