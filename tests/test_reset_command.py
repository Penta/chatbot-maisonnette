"""Tests de /reset. Lancer depuis la racine du projet : python -m unittest"""

import unittest
from unittest import mock

from bot.commands import reset_command

RIKA_CHANNEL, OTHER_CHANNEL = 10, 20


class ResetCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_reset_from_another_channel_empties_rika_channel(self):
        # Lancé depuis #général, /reset vide quand même la mémoire du salon de Rika, la seule qui existe.
        interaction = mock.Mock(channel_id=OTHER_CHANNEL)
        interaction.permissions.administrator = True
        interaction.response.defer = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        interaction.client.settings.channel_id = RIKA_CHANNEL
        interaction.client.history.reset = mock.AsyncMock()

        await reset_command.callback(interaction)

        interaction.client.history.reset.assert_awaited_once_with(RIKA_CHANNEL)
        self.assertIn(f"<#{RIKA_CHANNEL}>", interaction.followup.send.await_args.args[0])
