"""Tests du délai maximal d'attente (API_TIMEOUT). Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import anthropic

from bot.claude_client import ClaudeClient, ClaudeError


def fake_client(create) -> ClaudeClient:
    # Sans __init__ : pas de client Anthropic à créer.
    client = ClaudeClient.__new__(ClaudeClient)
    client.settings = SimpleNamespace(api_timeout=0.2)
    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client


class TimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_total_wait_is_capped(self):
        # Le SDK relancerait la requête après chaque délai dépassé : l'attente totale reste bornée.
        async def slow(**_):
            await asyncio.sleep(10)

        client = fake_client(slow)
        loop = asyncio.get_running_loop()
        start = loop.time()
        with self.assertLogs("chatbot.claude", "WARNING"), self.assertRaises(ClaudeError) as ctx:
            await client._call({})
        self.assertLess(loop.time() - start, 2)
        self.assertIn("trop de temps", ctx.exception.user_message)

    async def test_sdk_timeout_same_message(self):
        create = mock.AsyncMock(side_effect=anthropic.APITimeoutError(request=mock.Mock()))
        with self.assertLogs("chatbot.claude", "WARNING"), self.assertRaises(ClaudeError) as ctx:
            await fake_client(create)._call({})
        self.assertIn("trop de temps", ctx.exception.user_message)

    async def test_fast_answer_returned(self):
        create = mock.AsyncMock(return_value="réponse")
        self.assertEqual(await fake_client(create)._call({}), "réponse")


if __name__ == "__main__":
    unittest.main()
