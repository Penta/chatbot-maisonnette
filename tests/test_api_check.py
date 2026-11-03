"""Tests de la vérification de la clé API et du modèle au démarrage. Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import anthropic

from bot.claude_client import ClaudeClient
from bot.discord_bot import ChatBot


def fake_client(retrieve) -> ClaudeClient:
    # Sans __init__ : pas de client Anthropic à créer.
    client = ClaudeClient.__new__(ClaudeClient)
    client.settings = SimpleNamespace(api_timeout=0.2, model="claude-opus-5-5")
    client._client = SimpleNamespace(models=SimpleNamespace(retrieve=retrieve))
    return client


def api_error(cls, status: int):
    response = mock.Mock(status_code=status, headers={})
    return cls("erreur", response=response, body={"error": {"message": "détail de l'API"}})


class CheckAccessTest(unittest.IsolatedAsyncioTestCase):
    async def check(self, error):
        return await fake_client(mock.AsyncMock(side_effect=error)).check_access()

    async def test_valid_key_and_model(self):
        self.assertIsNone(await fake_client(mock.AsyncMock()).check_access())

    async def test_invalid_key(self):
        self.assertIn("ANTHROPIC_API_KEY", await self.check(api_error(anthropic.AuthenticationError, 401)))

    async def test_permission_denied(self):
        self.assertIn("détail de l'API", await self.check(api_error(anthropic.PermissionDeniedError, 403)))

    async def test_unknown_model(self):
        problem = await self.check(api_error(anthropic.NotFoundError, 404))
        self.assertIn("CLAUDE_MODEL", problem)
        self.assertIn("claude-opus-5-5", problem)

    async def test_api_down_does_not_block(self):
        # Panne passagère : on démarre quand même, avec un avertissement.
        for error in (api_error(anthropic.InternalServerError, 500),
                      anthropic.APIConnectionError(request=mock.Mock())):
            with self.subTest(error=type(error).__name__), self.assertLogs("chatbot.claude", "WARNING"):
                self.assertIsNone(await self.check(error))

    async def test_slow_api_does_not_block(self):
        async def slow(_):
            await asyncio.sleep(10)

        with self.assertLogs("chatbot.claude", "WARNING") as logs:
            self.assertIsNone(await fake_client(slow).check_access())
        self.assertIn("délai dépassé", logs.output[0])


class SetupHookTest(unittest.IsolatedAsyncioTestCase):
    async def test_bad_key_stops_bot(self):
        bot = mock.Mock(config_error=None)
        bot.claude.check_access = mock.AsyncMock(return_value="clé API Anthropic invalide")
        bot.close = mock.AsyncMock()
        with self.assertLogs("chatbot", "CRITICAL"):
            await ChatBot.setup_hook(bot)
        self.assertEqual(bot.config_error, "clé API Anthropic invalide")
        bot.close.assert_awaited_once()
        bot.tree.add_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
