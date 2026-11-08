"""Tests des repères de cache de prompt envoyés à Claude. Lancer depuis la racine du projet : python -m unittest"""

import unittest
from types import SimpleNamespace
from unittest import mock

from bot.claude_client import ClaudeClient
from bot.media import ImageData

EPHEMERAL = {"type": "ephemeral"}
IMAGE = ImageData("image/png", b"png")
HISTORY = [{"role": "user", "content": "Salut"}, {"role": "assistant", "content": "Coucou !"}]


class HistoryMarkerTest(unittest.IsolatedAsyncioTestCase):
    """La fin de l'historique porte son propre repère : le message suivant relit tout l'historique en cache,
    même si le nouveau message n'y est pas enregistré tel qu'il a été envoyé (image, GIF, fichier joint)."""

    def setUp(self):
        self.client = ClaudeClient.__new__(ClaudeClient)  # sans __init__ : pas de client Anthropic à créer
        self.client.settings = SimpleNamespace(
            model="modele", max_tokens=100, api_timeout=1, web_search=False, presence=False, thinking=False,
            effort=None, price_input_per_mtok=None, price_output_per_mtok=None, price_web_search_per_1000=10.0)
        self.client.memory = None
        self.client.personality = lambda: "Tu es Rika."
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, server_tool_use=None)
        response = SimpleNamespace(stop_reason="end_turn", model="modele", usage=usage,
                                   content=[SimpleNamespace(type="text", text="Jolie photo !")])
        self.create = mock.AsyncMock(return_value=response)
        self.client._client = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def sent(self) -> dict:
        return self.create.await_args.kwargs

    async def test_history_end_marked_before_an_image(self):
        history = [dict(message) for message in HISTORY]
        await self.client.ask(history, [("Regarde ça", [IMAGE])])
        messages = self.sent()["messages"]
        self.assertEqual(messages[1], {"role": "assistant", "content": [
            {"type": "text", "text": "Coucou !", "cache_control": EPHEMERAL}]})
        self.assertEqual(messages[0], HISTORY[0])
        self.assertEqual(history, HISTORY)  # l'historique reçu n'est pas modifié
        # Le nouveau message garde son repère : il sert aux relances de l'outil de mémoire.
        self.assertEqual(self.sent()["cache_control"], EPHEMERAL)

    async def test_history_end_marked_with_member_notes(self):
        await self.client.ask(list(HISTORY), [("Salut", [])], member_notes="Fiche de Bob")
        messages = self.sent()["messages"]
        self.assertEqual(messages[1]["content"][-1]["cache_control"], EPHEMERAL)
        self.assertEqual(messages[2]["content"], [
            {"type": "text", "text": "Salut", "cache_control": EPHEMERAL}, {"type": "text", "text": "Fiche de Bob"}])
        self.assertNotIn("cache_control", self.sent())

    async def test_no_history(self):
        await self.client.ask([], [("Salut", [])])
        self.assertEqual(self.sent()["messages"], [{"role": "user", "content": "Salut"}])

    async def test_nothing_cached_without_cache_history(self):
        # Mentions : contexte différent à chaque fois, rien ne sera relu.
        await self.client.ask(list(HISTORY), [("Salut", [])], cache_history=False)
        self.assertEqual(self.sent()["messages"][:2], HISTORY)
        self.assertNotIn("cache_control", self.sent())


if __name__ == "__main__":
    unittest.main()
