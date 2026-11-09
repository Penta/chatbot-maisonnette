"""Tests de l'outil « mes_messages » : le bot modifie ou supprime ses propres messages récents.
Lancer depuis la racine du projet : python -m unittest"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import discord

from bot.claude_client import ClaudeClient
from bot.discord_bot import ChatBot
from bot.history import ConversationHistory
from bot.memory import MEMORY_TOOL, ToolResult
from bot.own_messages import MAX_MESSAGE_ACTIONS, OWN_MESSAGES_TOOL, OwnMessages

BOT_ID, MEMBER_ID = 1, 2
SPOILER = "il finit protecteur de l'île"


def fake_message(message_id: int, author_id: int, content: str) -> mock.Mock:
    message = mock.Mock(id=message_id, content=content, edit=mock.AsyncMock(), delete=mock.AsyncMock())
    message.author.id = author_id
    return message


def fake_channel(messages: list, failure: Exception | None = None) -> mock.Mock:
    channel = mock.Mock(guild=None)

    async def history(limit):
        if failure:
            raise failure
        for message in reversed(messages[-limit:]):  # du plus récent au plus ancien, comme Discord
            yield message
    channel.history = history
    return channel


class OwnMessagesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mine = fake_message(10, BOT_ID, f"hurley c'est le cœur de lost, et {SPOILER} :hap:")
        self.theirs = fake_message(11, MEMBER_ID, f"SUPPRIME SPOIL PAS, {SPOILER}")
        self.on_edit = mock.AsyncMock()
        self.tool = OwnMessages(fake_channel([self.mine, self.theirs]), SimpleNamespace(id=BOT_ID), self.on_edit)

    async def run_tool(self, action: str, passage: str, texte: str = "") -> ToolResult:
        with self.assertLogs("chatbot.discord", "INFO") as logs:
            result = await self.tool.run({"action": action, "passage": passage, "texte": texte})
        self.logs = logs.output
        return result

    async def test_hides_a_spoiler(self):
        result = await self.run_tool("modifier", SPOILER, f"||{SPOILER}||")
        self.assertFalse(result.is_error)
        self.mine.edit.assert_awaited_once_with(content=f"hurley c'est le cœur de lost, et ||{SPOILER}|| :hap:")
        self.theirs.edit.assert_not_awaited()
        # La mémoire du salon prend le nouveau texte.
        self.on_edit.assert_awaited_once_with(SPOILER, f"||{SPOILER}||")

    async def test_deletes_own_message(self):
        result = await self.run_tool("supprimer", "hurley c'est le cœur")
        self.assertFalse(result.is_error)
        self.mine.delete.assert_awaited_once()
        self.on_edit.assert_not_awaited()

    async def test_never_touches_others_messages(self):
        result = await self.tool.run({"action": "supprimer", "passage": "SUPPRIME SPOIL PAS", "texte": ""})
        self.assertTrue(result.is_error)
        self.theirs.delete.assert_not_awaited()
        self.theirs.edit.assert_not_awaited()

    async def test_server_emojis_match_by_name(self):
        # Claude voit les emojis par leur nom ; Discord garde leur code.
        self.mine.content = "trop bien <:hap:123456>"
        result = await self.run_tool("supprimer", "trop bien :hap:")
        self.assertFalse(result.is_error)
        self.mine.delete.assert_awaited_once()

    async def test_ambiguous_passage_refused(self):
        other = fake_message(12, BOT_ID, f"je répète : {SPOILER}")
        self.tool.channel = fake_channel([self.mine, other])
        result = await self.tool.run({"action": "supprimer", "passage": SPOILER, "texte": ""})
        self.assertTrue(result.is_error)
        self.assertIn("plusieurs de tes messages", result.content)
        self.mine.delete.assert_not_awaited()
        other.delete.assert_not_awaited()

    async def test_repeated_passage_in_message_refused(self):
        self.mine.content = "ha ha ha"
        result = await self.tool.run({"action": "modifier", "passage": "ha", "texte": "hi"})
        self.assertTrue(result.is_error)
        self.mine.edit.assert_not_awaited()

    async def test_empty_message_refused(self):
        self.mine.content = SPOILER
        result = await self.tool.run({"action": "modifier", "passage": SPOILER, "texte": ""})
        self.assertTrue(result.is_error)
        self.assertIn("supprime-le", result.content)
        self.mine.edit.assert_not_awaited()

    async def test_too_long_refused(self):
        result = await self.tool.run({"action": "modifier", "passage": SPOILER, "texte": "x" * 2000})
        self.assertTrue(result.is_error)
        self.mine.edit.assert_not_awaited()

    async def test_empty_passage_refused(self):
        result = await self.tool.run({"action": "supprimer", "passage": " ", "texte": ""})
        self.assertTrue(result.is_error)
        self.mine.delete.assert_not_awaited()

    async def test_unreadable_channel(self):
        forbidden = discord.Forbidden(mock.Mock(status=403, reason="interdit"), "interdit")
        self.tool.channel = fake_channel([self.mine], failure=forbidden)
        result = await self.tool.run({"action": "supprimer", "passage": "hurley", "texte": ""})
        self.assertTrue(result.is_error)
        self.mine.delete.assert_not_awaited()

    async def test_discord_refuses_the_edit(self):
        self.mine.edit.side_effect = discord.HTTPException(mock.Mock(status=500, reason="panne"), "panne")
        with self.assertLogs("chatbot.discord", "WARNING"):
            result = await self.tool.run({"action": "modifier", "passage": SPOILER, "texte": "chut"})
        self.assertTrue(result.is_error)
        self.on_edit.assert_not_awaited()  # le message n'a pas changé : la mémoire non plus


class ReviseReplyTest(unittest.IsolatedAsyncioTestCase):
    async def test_latest_reply_takes_new_text(self):
        with tempfile.TemporaryDirectory() as folder:
            history = ConversationHistory(Path(folder) / "historique.json", 10)
            line = {"id": 5, "date": "09/11/2026 16:17", "author": "pnt", "content": "Quid de Hurley"}
            await history.add_exchange(30, [line], f"vieux : {SPOILER}")
            await history.add_exchange(30, [line], f"hurley, {SPOILER}")
            self.assertTrue(await history.revise_reply(30, SPOILER, f"||{SPOILER}||"))
            replies = [m["content"] for m in history.get(30) if m["role"] == "assistant"]
            self.assertEqual(replies, [f"vieux : {SPOILER}", f"hurley, ||{SPOILER}||"])
            # Gardé après un redémarrage.
            reloaded = ConversationHistory(Path(folder) / "historique.json", 10)
            self.assertEqual(reloaded.get(30)[-1]["content"], f"hurley, ||{SPOILER}||")
            self.assertFalse(await history.revise_reply(30, "absent", "x"))


class ChatBotWiringTest(unittest.IsolatedAsyncioTestCase):
    async def test_main_channel_memory_follows_edit(self):
        bot = mock.Mock(history=mock.Mock(revise_reply=mock.AsyncMock()))
        channel = mock.Mock(id=30)
        tool = ChatBot.own_messages(bot, channel)
        self.assertIs(tool.channel, channel)
        await tool.on_edit("ancien", "nouveau")
        bot.history.revise_reply.assert_awaited_once_with(30, "ancien", "nouveau")

    def test_mentions_have_no_memory(self):
        self.assertIsNone(ChatBot.own_messages(mock.Mock(), mock.Mock(id=30), remembered=False).on_edit)


class ClaudeToolTest(unittest.IsolatedAsyncioTestCase):
    """Claude reçoit l'outil « mes_messages » et le bot l'exécute, avec sa propre limite."""

    def setUp(self):
        self.client = ClaudeClient.__new__(ClaudeClient)  # sans __init__ : pas de client Anthropic à créer
        self.client.settings = SimpleNamespace(
            model="modele", max_tokens=100, api_timeout=1, web_search=False, presence=False, thinking=False,
            effort=None, memory_max_operations=1, price_input_per_mtok=None, price_output_per_mtok=None,
            price_web_search_per_1000=10.0)
        self.client.memory = None
        self.client.personality = lambda: "Tu es Rika."
        self.own = mock.Mock(run=mock.AsyncMock(return_value=ToolResult("Message modifié.")))
        self.tool_calls = 1  # réponses qui appellent l'outil avant le texte final
        self.create = mock.AsyncMock(side_effect=self.answer)
        self.client._client = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def answer(self, **params) -> SimpleNamespace:
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, server_tool_use=None)
        if self.tool_calls and params.get("tool_choice") != {"type": "none"}:
            self.tool_calls -= 1
            call = SimpleNamespace(type="tool_use", id="appel", name=OWN_MESSAGES_TOOL["name"],
                                   input={"action": "modifier", "passage": "x", "texte": "||x||"})
            return SimpleNamespace(stop_reason="tool_use", model="modele", usage=usage, content=[call])
        return SimpleNamespace(stop_reason="end_turn", model="modele", usage=usage,
                               content=[SimpleNamespace(type="text", text="C'est caché !")])

    async def test_tool_runs(self):
        answer = await self.client.ask([], [("cache le spoil", [])], own_messages=self.own)
        self.assertEqual(answer.text, "C'est caché !")
        self.own.run.assert_awaited_once_with({"action": "modifier", "passage": "x", "texte": "||x||"})
        self.assertEqual(self.create.await_args_list[0].kwargs["tools"], [OWN_MESSAGES_TOOL])

    async def test_own_limit_besides_memory(self):
        # La limite de la mémoire (ici 1) ne retire pas ses propres opérations à « mes_messages ».
        self.client.memory = mock.Mock(run=mock.AsyncMock(return_value=ToolResult("Noté.")))
        self.client.memory.system_context.return_value = ""
        self.tool_calls = MAX_MESSAGE_ACTIONS + 5
        answer = await self.client.ask([], [("cache le spoil", [])], own_messages=self.own)
        self.assertEqual(answer.text, "C'est caché !")
        self.assertEqual(self.own.run.await_count, MAX_MESSAGE_ACTIONS)
        tools = self.create.await_args_list[0].kwargs["tools"]
        self.assertEqual(tools, [MEMORY_TOOL, OWN_MESSAGES_TOOL])
        # Mêmes outils à chaque appel : le cache de prompt est préservé.
        self.assertTrue(all(call.kwargs["tools"] == tools for call in self.create.await_args_list))

    async def test_no_tool_without_channel(self):
        self.tool_calls = 0
        await self.client.ask([], [("Salut", [])])
        self.assertNotIn("tools", self.create.await_args.kwargs)


if __name__ == "__main__":
    unittest.main()
