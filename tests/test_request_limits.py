"""Tests du budget de pièces jointes par requête : des messages arrivés pendant une réponse, trop lourds
ensemble, partent dans plusieurs appels à Claude. Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import base64
import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import anthropic

from bot.claude_client import LIMIT_NOTE, ClaudeClient, ClaudeError
from bot.discord_bot import ChatBot, PendingMessage
from bot.media import PDF_MEDIA_TYPE, ImageData, base64_size, fit_in_request
from bot.memory import MEMORY_TOOL, ToolResult

CHANNEL_ID = 10


def fake_bot() -> mock.Mock:
    bot = mock.Mock()
    bot._pending = {}
    bot._preparing = {}
    bot._deleted = set()
    bot.history.lock.return_value = asyncio.Lock()
    bot.history.get.return_value = []
    bot.history.add_exchange = mock.AsyncMock()
    bot.presence_note.return_value = None
    bot.member_notes = mock.AsyncMock(return_value="")
    bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text="Réponse", web_used=False))
    bot.send_answer = mock.AsyncMock()
    bot.safe_reply = mock.AsyncMock()
    return bot


def pdf(megabytes: float, pages: int = 1) -> ImageData:
    return ImageData(PDF_MEDIA_TYPE, bytes(int(megabytes * 1_000_000)), name="doc.pdf", pages=pages)


def image() -> ImageData:
    return ImageData("image/png", bytes(1000))


def pending(message_id: int, channel: mock.Mock, images: list[ImageData]) -> PendingMessage:
    message = mock.Mock(id=message_id, channel=channel)
    line = {"id": message_id, "date": "03/11/2026 18:00", "author": f"membre{message_id}",
            "content": f"message {message_id}"}
    return PendingMessage(message, line, "", images)


def asked_texts(bot: mock.Mock) -> list[list[str]]:
    """Textes envoyés à chaque appel à Claude."""
    return [[text for text, _ in call.args[1]] for call in bot.claude.ask.await_args_list]


def text(message_id: int) -> str:
    return f"03/11/2026 18:00, membre{message_id} a écrit : message {message_id}"


class BudgetTest(unittest.TestCase):
    def test_base64_size(self):
        for size in (0, 1, 2, 3, 4, 1000, 15_000_001):
            with self.subTest(size=size):
                self.assertEqual(base64_size(size), len(base64.standard_b64encode(bytes(size))))

    def test_light_messages_together(self):
        self.assertEqual(fit_in_request([[pdf(5)], [image(), image()], [], [pdf(5)]]), 4)

    def test_heavy_messages_apart(self):
        # 15 Mo bruts = 20 Mo en base64 : deux ne tiennent pas dans les 24 Mo du budget.
        self.assertEqual(fit_in_request([[pdf(15)], [pdf(15)], [pdf(1)]]), 1)
        self.assertEqual(fit_in_request([[pdf(15)], [pdf(1)], [pdf(15)]]), 2)

    def test_first_message_always_goes(self):
        # Un PDF de 20 Mo (26,7 Mo en base64) dépasse le budget, mais part quand même, seul.
        self.assertEqual(fit_in_request([[pdf(20)]]), 1)
        self.assertEqual(fit_in_request([[pdf(20)], []]), 1)

    def test_page_limit(self):
        self.assertEqual(fit_in_request([[pdf(1, pages=40)], [pdf(1, pages=40)], [pdf(1, pages=40)]]), 2)
        self.assertEqual(fit_in_request([[pdf(1, pages=100)], [pdf(1, pages=1)]]), 1)
        # Une image compte pour une page : 100 au plus par requête.
        self.assertEqual(fit_in_request([[image()] * 4] * 30), 25)

    def test_text_limit(self):
        # 100 Ko : la taille maximale d'un fichier texte joint. Le premier message part toujours.
        big = "x" * 100_000
        self.assertEqual(fit_in_request([[]] * 4, [big] * 4), 3)
        self.assertEqual(fit_in_request([[], []], ["x" * 400_000, "court"]), 1)
        self.assertEqual(fit_in_request([[]] * 3, ["court"] * 3), 3)


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class MainChannelTest(unittest.IsolatedAsyncioTestCase):
    async def test_heavy_messages_split(self):
        # Deux PDF de 15 Mo (40 Mo en base64) : trop lourds pour partir dans la même requête.
        bot = fake_bot()
        channel = mock.Mock(id=CHANNEL_ID)
        bot._pending[CHANNEL_ID] = [pending(1, channel, [pdf(15)]), pending(2, channel, [pdf(15)])]
        await ChatBot.answer_main_channel(bot, channel)
        self.assertEqual(asked_texts(bot), [[text(1)], [text(2)]])
        self.assertEqual(bot.history.add_exchange.await_count, 2)
        self.assertEqual([call.args[0].id for call in bot.send_answer.await_args_list], [1, 2])
        # Réponse découpée : chacune cite son message, pour qu'on sache à quoi elle répond.
        self.assertEqual([call.kwargs["reply"] for call in bot.send_answer.await_args_list], [True, True])
        self.assertEqual(bot._pending[CHANNEL_ID], [])

    async def test_light_messages_grouped(self):
        bot = fake_bot()
        channel = mock.Mock(id=CHANNEL_ID)
        bot._pending[CHANNEL_ID] = [pending(2, channel, [pdf(5)]), pending(1, channel, [pdf(5)])]
        await ChatBot.answer_main_channel(bot, channel)
        self.assertEqual(asked_texts(bot), [[text(1), text(2)]])
        bot.send_answer.assert_awaited_once_with(mock.ANY, mock.ANY, reply=False)

    async def test_page_limit(self):
        # 3 PDF de 40 pages : les deux premiers ensemble (80 pages), le troisième ensuite.
        bot = fake_bot()
        channel = mock.Mock(id=CHANNEL_ID)
        bot._pending[CHANNEL_ID] = [pending(i, channel, [pdf(1, pages=40)]) for i in (1, 2, 3)]
        await ChatBot.answer_main_channel(bot, channel)
        self.assertEqual(asked_texts(bot), [[text(1), text(2)], [text(3)]])

    async def test_error_does_not_stop_the_rest(self):
        bot = fake_bot()
        channel = mock.Mock(id=CHANNEL_ID)
        first, second = pending(1, channel, [pdf(15)]), pending(2, channel, [pdf(15)])
        bot._pending[CHANNEL_ID] = [first, second]
        bot.claude.ask.side_effect = [ClaudeError("Erreur."), mock.Mock(text="Réponse", web_used=False)]
        await ChatBot.answer_main_channel(bot, channel)
        bot.safe_reply.assert_awaited_once_with(first.message, "Erreur.")
        bot.send_answer.assert_awaited_once_with(second.message, "Réponse", reply=True)
        bot.history.add_exchange.assert_awaited_once()
        self.assertEqual(bot.forget_waiting.call_args_list, [mock.call(first.message), mock.call(second.message)])

    async def test_long_text_files_split(self):
        # Quatre fichiers texte de 100 Ko, sans image ni PDF : trop longs pour partir dans la même requête.
        bot = fake_bot()
        channel = mock.Mock(id=CHANNEL_ID)
        messages = [pending(i, channel, []) for i in (1, 2, 3, 4)]
        for waiting in messages:
            waiting.extra = "x" * 100_000
        bot._pending[CHANNEL_ID] = messages
        await ChatBot.answer_main_channel(bot, channel)
        self.assertEqual([len(call.args[1]) for call in bot.claude.ask.await_args_list], [3, 1])
        self.assertEqual([call.args[0].id for call in bot.send_answer.await_args_list], [3, 4])

    async def test_error_warns_every_author(self):
        # Deux messages d'Alice et un de Bob : chacun est prévenu, sur son dernier message du lot.
        bot = fake_bot()
        channel = mock.Mock(id=CHANNEL_ID)
        batch = [pending(i, channel, []) for i in (1, 2, 3)]
        batch[0].message.author.id = batch[2].message.author.id = 100
        batch[1].message.author.id = 200
        bot._pending[CHANNEL_ID] = list(batch)
        bot.claude.ask.side_effect = ClaudeError("Erreur.")
        await ChatBot.answer_main_channel(bot, channel)
        replied = {call.args[0].id: call.args[1] for call in bot.safe_reply.await_args_list}
        self.assertEqual(replied, {2: "Erreur.", 3: "Erreur."})
        self.assertEqual(bot.safe_reply.await_count, 2)
        bot.history.add_exchange.assert_not_awaited()

    async def test_deferred_answer_handles_all(self):
        # Absent : une seule tâche attend pour tous les messages du salon, et doit répondre à chacun.
        bot = fake_bot()
        bot._waiters = {}
        bot.presence.wait_turn = mock.AsyncMock()
        bot.presence.late_note.return_value = "[en retard]"
        bot.answer_main_channel = lambda channel, late=None: ChatBot.answer_main_channel(bot, channel, late)
        bot._answer_when_possible = lambda *args: ChatBot._answer_when_possible(bot, *args)
        channel = mock.Mock(id=CHANNEL_ID)
        for message_id in (1, 2, 3):
            waiting = pending(message_id, channel, [pdf(15)])
            bot._pending.setdefault(CHANNEL_ID, []).append(waiting)
            ChatBot.answer_later(bot, waiting.message, bot.answer_main_channel)
        await bot._waiters[CHANNEL_ID]

        # Chaque réponse rappelle qu'elle arrive en retard, et la garde en mémoire.
        self.assertEqual(asked_texts(bot), [["[en retard]", text(i)] for i in (1, 2, 3)])
        for call in bot.history.add_exchange.await_args_list:
            self.assertEqual(call.args[1][0], {"content": "[en retard]"})
        self.assertEqual([call.args[0].id for call in bot.send_answer.await_args_list], [1, 2, 3])
        self.assertEqual(bot._pending[CHANNEL_ID], [])
        self.assertEqual(bot._waiters, {})


class TooLargeTest(unittest.IsolatedAsyncioTestCase):
    async def test_clear_message(self):
        error = anthropic.RequestTooLargeError("trop lourd", response=mock.Mock(status_code=413, headers={}),
                                               body={"error": {"type": "request_too_large"}})
        client = ClaudeClient.__new__(ClaudeClient)  # sans __init__ : pas de client Anthropic à créer
        client.settings = SimpleNamespace(api_timeout=1)
        client._client = SimpleNamespace(messages=SimpleNamespace(create=mock.AsyncMock(side_effect=error)))
        with self.assertLogs("chatbot.claude", "ERROR"), self.assertRaises(ClaudeError) as ctx:
            await client._call({}, {"pdf"})
        self.assertEqual(ctx.exception.user_message,
                         "Ces fichiers sont trop lourds pour moi tous ensemble, renvoie-les un par un.")


class WebToolsTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def response(stop_reason: str) -> SimpleNamespace:
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, server_tool_use=None)
        return SimpleNamespace(stop_reason=stop_reason, model="modele", usage=usage,
                               content=[SimpleNamespace(type="text", text="Réponse")])

    async def test_limits_hold_for_the_whole_message(self):
        # L'API ne compte d'un appel à l'autre que les appels directs : sans allowed_callers, ceux lancés
        # depuis le filtrage automatique repartiraient de zéro à chaque relance.
        client = ClaudeClient.__new__(ClaudeClient)  # sans __init__ : pas de client Anthropic à créer
        client.settings = SimpleNamespace(
            model="modele", max_tokens=100, api_timeout=1, web_search=True, web_max_uses=3, presence=False,
            thinking=False, effort=None, price_input_per_mtok=None, price_output_per_mtok=None,
            price_web_search_per_1000=10.0)
        client.memory = None
        client.personality = lambda: "Tu es Rika."
        create = mock.AsyncMock(side_effect=[self.response("pause_turn"), self.response("end_turn")])
        client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
        await client.ask([], [("Bonjour", [])])
        first, second = (call.kwargs["tools"] for call in create.await_args_list)
        self.assertEqual(first, [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": 3, "allowed_callers": ["direct"]},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3, "allowed_callers": ["direct"],
             "max_content_tokens": 10_000},
        ])
        # Mêmes outils à chaque appel : l'API l'exige pour reprendre, et le cache de prompt est préservé.
        self.assertEqual(second, first)


def limit_notes(params: dict) -> list[str]:
    """Textes ajoutés par le bot après les résultats d'outils du dernier message (avis de limite atteinte)."""
    content = params["messages"][-1]["content"]
    if not isinstance(content, list):
        return []
    return [block["text"] for block in content if block.get("type") == "text"]


class MemoryLimitTest(unittest.IsolatedAsyncioTestCase):
    """Modèle qui rappelle l'outil « memoire » tant que l'API le lui permet (ou, avec obeys, jusqu'à
    ce que le bot lui signale la limite)."""

    def setUp(self):
        self.client = ClaudeClient.__new__(ClaudeClient)  # sans __init__ : pas de client Anthropic à créer
        self.client.settings = SimpleNamespace(
            model="modele", max_tokens=100, api_timeout=1, web_search=False, presence=False, thinking=False,
            effort=None, memory_max_operations=2, price_input_per_mtok=None, price_output_per_mtok=None,
            price_web_search_per_1000=10.0)
        self.client.memory = mock.Mock(run=mock.AsyncMock(return_value=ToolResult("Noté.")))
        self.client.memory.system_context.return_value = ""
        self.client.personality = lambda: "Tu es Rika."
        self.calls_per_response = 1
        self.obeys = False  # s'arrête de noter dès que le bot lui signale la limite
        self.create = mock.AsyncMock(side_effect=self.answer)
        self.client._client = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def answer(self, **params) -> SimpleNamespace:
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, server_tool_use=None)
        if params.get("tool_choice") == {"type": "none"} or (self.obeys and limit_notes(params)):
            return SimpleNamespace(stop_reason="end_turn", model="modele", usage=usage,
                                   content=[SimpleNamespace(type="text", text="Enchantée !")])
        calls = [SimpleNamespace(type="tool_use", id=f"appel{i}", name=MEMORY_TOOL["name"], input={"action": "ajouter"})
                 for i in range(self.calls_per_response)]
        return SimpleNamespace(stop_reason="tool_use", model="modele", usage=usage, content=calls)

    async def test_limit_keeps_cache(self):
        # Changer tool_choice en cours de réponse ferait relire toute la conversation au prix fort : le bot
        # signale plutôt la limite à Claude, qui répond alors sans outil.
        self.obeys = True
        answer = await self.client.ask([], [("J'ai trois chats.", [])])
        self.assertEqual(answer.text, "Enchantée !")
        self.assertEqual(self.client.memory.run.await_count, 2)
        choices = [call.kwargs.get("tool_choice") for call in self.create.await_args_list]
        self.assertEqual(choices, [None, None, None])
        notes = [limit_notes(call.kwargs) for call in self.create.await_args_list[1:]]
        self.assertEqual(notes, [[], [LIMIT_NOTE.format(name=MEMORY_TOOL["name"])]])
        # La liste des outils ne change pas : retirer un outil ferait réécrire tout le cache de prompt.
        tools = [call.kwargs["tools"] for call in self.create.await_args_list]
        self.assertEqual(tools, [[MEMORY_TOOL]] * 3)

    async def test_text_answer_if_limit_ignored(self):
        # Claude rappelle quand même l'outil : refusé, puis réponse forcée en texte, une seule fois.
        answer = await self.client.ask([], [("J'ai trois chats.", [])])
        self.assertEqual(answer.text, "Enchantée !")
        self.assertEqual(self.client.memory.run.await_count, 2)
        choices = [call.kwargs.get("tool_choice") for call in self.create.await_args_list]
        self.assertEqual(choices, [None, None, None, {"type": "none"}])
        results = self.create.await_args_list[-1].kwargs["messages"][-1]["content"]
        self.assertEqual([result["is_error"] for result in results], [True])

    async def test_calls_beyond_limit_in_one_response_are_refused(self):
        self.calls_per_response = 3
        self.obeys = True
        answer = await self.client.ask([], [("J'ai trois chats.", [])])
        self.assertEqual(answer.text, "Enchantée !")
        self.assertEqual(self.client.memory.run.await_count, 2)
        results = self.create.await_args_list[-1].kwargs["messages"][-1]["content"]
        self.assertEqual([result.get("is_error") for result in results], [False, False, True, None])
        self.assertNotIn("tool_choice", self.create.await_args_list[-1].kwargs)


if __name__ == "__main__":
    unittest.main()
