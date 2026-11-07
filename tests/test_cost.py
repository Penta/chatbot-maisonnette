"""Tests du coût estimé dans le journal. Lancer depuis la racine du projet : python -m unittest"""

import unittest
from types import SimpleNamespace

from bot.claude_client import ClaudeClient


def logged_cost(model: str, price_input: float, price_output: float, **usage) -> str:
    """Coût estimé écrit dans le journal pour une réponse avec cette consommation."""
    # Sans __init__ : pas de client Anthropic à créer.
    client = ClaudeClient.__new__(ClaudeClient)
    client.settings = SimpleNamespace(model=model, price_input_per_mtok=price_input,
                                      price_output_per_mtok=price_output, price_web_search_per_1000=10.0)
    tokens = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
              "output_tokens": 0, "server_tool_use": None, **usage}
    response = SimpleNamespace(model=model, usage=SimpleNamespace(**tokens))
    with unittest.TestCase().assertLogs("chatbot.claude", level="INFO") as logs:
        client._log_usage([response])
    return logs.output[-1].rsplit("coût estimé : ", 1)[1]


class CacheReadCostTest(unittest.TestCase):
    def test_opus_5_5_cache_read_at_5_percent(self):
        # 1 million de tokens lus en cache, à 4 $ le million : 0,20 $ sur Opus 5.5.
        self.assertEqual(logged_cost("claude-opus-5-5", 4.0, 20.0, cache_read_input_tokens=1_000_000),
                         "0.20000 $")

    def test_sonnet_5_5_cache_read_at_10_percent(self):
        self.assertEqual(logged_cost("claude-sonnet-5-5", 2.0, 10.0, cache_read_input_tokens=1_000_000),
                         "0.20000 $")

    def test_unknown_model_cache_read_at_10_percent(self):
        # Modèle absent de la table, tarifs donnés dans le .env : 10 %, le cas le plus courant.
        self.assertEqual(logged_cost("claude-futur-9", 3.0, 15.0, cache_read_input_tokens=1_000_000),
                         "0.30000 $")

    def test_other_tokens_unchanged(self):
        # Entrée normale, écriture en cache (125 %) et sortie : 1 + 1,25 + 2 $.
        self.assertEqual(logged_cost("claude-opus-5-5", 4.0, 20.0, input_tokens=250_000,
                                     cache_creation_input_tokens=250_000, output_tokens=100_000),
                         "4.25000 $")


if __name__ == "__main__":
    unittest.main()
