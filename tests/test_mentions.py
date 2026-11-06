"""Tests des mentions hors du salon dédié : une rafale ne lance pas un appel par message.
Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import contextlib
import unittest
from unittest import mock

from bot.claude_client import ClaudeError
from bot.discord_bot import ChatBot
from bot.media import PDF_MEDIA_TYPE, ImageData

CHANNEL_ID = 20


def fake_bot() -> mock.Mock:
    bot = mock.Mock()
    bot._mentions = {}
    bot._waiters = {}
    bot.available.return_value = True
    lock = asyncio.Lock()
    bot.history.lock.return_value = lock
    bot.answer_mentions = lambda channel, late=None: ChatBot.answer_mentions(bot, channel, late)
    bot.answer_mention = lambda mentions, late=None: ChatBot.answer_mention(bot, mentions, late)
    bot.mention_part = lambda message: ChatBot.mention_part(bot, message)
    bot.mention_context = lambda channel, ids, before=None: ChatBot.mention_context(bot, channel, ids, before)
    bot.settings.context_message_limit = 20
    bot.settings.pdf_max_pages = 100
    bot.presence_note.return_value = None
    bot.timestamp = lambda message: "03/11/2026 18:00"
    bot.read_reply = mock.AsyncMock(return_value=(None, []))
    bot.member_notes = mock.AsyncMock(return_value="")
    bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text="Coucou"))
    bot.send_answer = mock.AsyncMock()
    bot.safe_reply = mock.AsyncMock()
    return bot


def fake_mention(message_id: int, text: str, channel: mock.Mock) -> mock.Mock:
    message = mock.Mock(id=message_id, clean_content=f"@Rika {text}", attachments=[], stickers=[])
    message.channel = channel
    message.guild.me.display_name = "Rika"
    message.author.display_name = message.author.name = f"membre{message_id}"
    return message


def fake_channel(history: list) -> mock.Mock:
    channel = mock.Mock(id=CHANNEL_ID)
    channel.name = "général"

    async def read(limit, before):
        for msg in reversed([m for m in history if before is None or m.id < before.id][-limit:]):
            yield msg
    channel.history = read
    return channel


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class MentionTest(unittest.IsolatedAsyncioTestCase):
    async def test_burst_grouped(self):
        # 10 mentions en rafale : la première part seule, les 9 autres attendent puis partent ensemble.
        bot = fake_bot()
        calls, release = [], asyncio.Event()

        async def answer(mentions, late=None):
            calls.append([m.id for m in mentions])
            await release.wait()
        bot.answer_mention = answer
        channel = mock.Mock(id=CHANNEL_ID)
        tasks = [asyncio.create_task(ChatBot.handle_mention(bot, mock.Mock(id=1, channel=channel, attachments=[])))]
        await asyncio.sleep(0)
        tasks += [asyncio.create_task(ChatBot.handle_mention(bot, mock.Mock(id=i, channel=channel, attachments=[])))
                  for i in range(2, 11)]
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(calls, [[1], list(range(2, 11))])
        self.assertEqual(bot._mentions, {})

    async def test_one_call_for_grouped_mentions(self):
        bot = fake_bot()
        history = []
        channel = fake_channel(history)
        history += [mock.Mock(id=1, clean_content="salut tout le monde", attachments=[], stickers=[]),
                    fake_mention(2, "ça va ?", channel),
                    mock.Mock(id=3, clean_content="il dort", attachments=[], stickers=[]),
                    fake_mention(4, "réveille-toi", channel)]
        for msg in history:
            msg.author.display_name = msg.author.name = f"membre{msg.id}"
        bot._mentions[CHANNEL_ID] = [history[3], history[1]]
        await ChatBot.answer_mentions(bot, channel)

        bot.claude.ask.assert_awaited_once()
        parts = bot.claude.ask.await_args.args[1]
        self.assertEqual(len(parts), 2)  # une partie par mention, dans l'ordre
        context, second = parts[0][0], parts[1][0]
        self.assertIn("salut tout le monde", context)
        self.assertIn("il dort", context)
        self.assertNotIn("membre2 : ", context)  # la mention n'est pas recopiée dans le contexte
        self.assertTrue(context.endswith("membre2 te mentionne et écrit : ça va ?"))
        self.assertEqual(second, "03/11/2026 18:00, membre4 te mentionne et écrit : réveille-toi")
        bot.member_notes.assert_awaited_once_with([history[1], history[3]])
        bot.send_answer.assert_awaited_once_with(history[3], "Coucou", reply=False)

    async def test_grouped_answer_sees_previous_answer(self):
        # La réponse à la 1re mention est postée après les mentions regroupées : elle doit être dans le contexte.
        bot = fake_bot()
        history = []
        channel = fake_channel(history)
        history += [fake_mention(1, "tu fais quoi ce soir ?", channel), fake_mention(2, "?", channel),
                    fake_mention(3, "réponds !", channel),
                    mock.Mock(id=4, clean_content="Je regarde un film !", attachments=[], stickers=[])]
        history[3].author.display_name = history[3].author.name = "Rika"
        await ChatBot.answer_mention(bot, history[1:3])
        self.assertIn("Rika : Je regarde un film !", bot.claude.ask.await_args.args[1][0][0])

    async def test_late_context_stops_at_last_mention(self):
        # En retard, la discussion qui a suivi les mentions ne remplace pas celle qui les entoure.
        bot = fake_bot()
        history = []
        channel = fake_channel(history)
        history += [mock.Mock(id=1, clean_content="avant", attachments=[], stickers=[]),
                    fake_mention(2, "tu es là ?", channel),
                    mock.Mock(id=3, clean_content="après", attachments=[], stickers=[])]
        for msg in history:
            msg.author.display_name = msg.author.name = f"membre{msg.id}"
        bot.presence.late_note.return_value = "(en retard)"
        await ChatBot.answer_mention(bot, [history[1]], late=mock.Mock())
        context = bot.claude.ask.await_args.args[1][0][0]
        self.assertIn("avant", context)
        self.assertNotIn("après", context)

    async def test_refused_attachment_drops_only_that_mention(self):
        bot = fake_bot()
        channel = fake_channel([])
        first, second = fake_mention(1, "regarde", channel), fake_mention(2, "et ça ?", channel)
        bot.mention_part = mock.AsyncMock(side_effect=[None, ("texte", [])])
        await ChatBot.answer_mention(bot, [first, second])
        self.assertEqual(bot.claude.ask.await_args.args[1], [("texte", [])])
        bot.member_notes.assert_awaited_once_with([second])

    async def test_error_warns_every_author(self):
        bot = fake_bot()
        channel = fake_channel([])
        first, second = fake_mention(1, "salut", channel), fake_mention(2, "coucou", channel)
        bot.claude.ask.side_effect = ClaudeError("Erreur.")
        await ChatBot.answer_mention(bot, [first, second])
        self.assertCountEqual(bot.safe_reply.await_args_list, [mock.call(first, "Erreur."), mock.call(second, "Erreur.")])

    async def test_long_text_files_split(self):
        # Quatre mentions avec un fichier texte de 100 Ko : trop longues pour partir dans la même requête.
        bot = fake_bot()
        channel = fake_channel([])
        mentions = [fake_mention(i, "lis ça", channel) for i in (1, 2, 3, 4)]
        bot.mention_part = mock.AsyncMock(return_value=("x" * 100_000, []))
        await ChatBot.answer_mention(bot, mentions)
        self.assertEqual([len(call.args[1]) for call in bot.claude.ask.await_args_list], [3, 1])

    async def test_heavy_mentions_split(self):
        # Deux mentions avec chacune un PDF de 15 Mo : trop lourdes pour une seule requête, elles reçoivent
        # deux réponses à la suite, et la seconde voit la première dans le contexte.
        bot = fake_bot()
        history = []
        channel = fake_channel(history)
        history += [mock.Mock(id=1, clean_content="salut", attachments=[], stickers=[]),
                    fake_mention(2, "lis ça", channel), fake_mention(3, "et ça", channel)]
        history[0].author.display_name = history[0].author.name = "membre1"
        heavy = ImageData(PDF_MEDIA_TYPE, bytes(15_000_000), name="doc.pdf")
        bot.mention_part = mock.AsyncMock(side_effect=[("lis ça", [heavy]), ("et ça", [heavy])])

        async def post(message, text, reply=False):
            answer = mock.Mock(id=len(history) + 1, clean_content=text, attachments=[], stickers=[])
            answer.author.display_name = answer.author.name = "Rika"
            history.append(answer)
        bot.send_answer = mock.AsyncMock(side_effect=post)
        await ChatBot.answer_mention(bot, history[1:3])

        first, second = (call.args[1] for call in bot.claude.ask.await_args_list)
        self.assertEqual([len(first), len(second)], [1, 1])
        self.assertIn("membre1 : salut", first[0][0])
        self.assertTrue(first[0][0].endswith("lis ça"))
        self.assertNotIn("Rika : Coucou", first[0][0])
        self.assertIn("Rika : Coucou", second[0][0])
        self.assertTrue(second[0][0].endswith("et ça"))
        self.assertEqual(bot.member_notes.await_args_list, [mock.call([history[1]]), mock.call([history[2]])])
        self.assertEqual(bot.send_answer.await_args_list,
                         [mock.call(history[1], "Coucou", reply=True), mock.call(history[2], "Coucou", reply=True)])

    def heavy_split_setup(self, late: bool):
        """Deux mentions trop lourdes pour une seule requête, suivies d'un message d'un membre ;
        chaque réponse du bot est ajoutée au salon."""
        bot = fake_bot()
        history = []
        channel = fake_channel(history)
        history += [fake_mention(1, "lis ça", channel), fake_mention(2, "et ça", channel)]
        heavy = ImageData(PDF_MEDIA_TYPE, bytes(15_000_000), name="doc.pdf")
        bot.mention_part = mock.AsyncMock(side_effect=[("lis ça", [heavy]), ("et ça", [heavy])])
        bot.presence.late_note.return_value = "(en retard)"
        for message in history:
            message.guild.me.display_name = message.guild.me.name = "Rika"
        if late:
            history.append(mock.Mock(id=3, clean_content="après", attachments=[], stickers=[]))
            history[2].author.display_name = history[2].author.name = "membre3"
        answers = iter(["Réponse un", "Réponse deux"])
        bot.claude.ask = mock.AsyncMock(side_effect=lambda *a, **k: mock.Mock(text=next(answers)))

        async def post(message, text, reply=False):
            answer = mock.Mock(id=len(history) + 1, clean_content=text, attachments=[], stickers=[])
            answer.author.display_name = answer.author.name = "Rika"
            history.append(answer)
        bot.send_answer = mock.AsyncMock(side_effect=post)
        return bot, history[:2]

    async def test_heavy_split_next_group_sees_answered_mention(self):
        # La 2e réponse voit la 1re mention (déjà traitée) comme un message normal, et la réponse du bot.
        bot, mentions = self.heavy_split_setup(late=False)
        await ChatBot.answer_mention(bot, mentions)
        first, second = (call.args[1][0][0] for call in bot.claude.ask.await_args_list)
        self.assertNotIn("membre1 : ", first)
        self.assertNotIn("membre2 : ", first)
        self.assertIn("membre1 : @Rika lis ça", second)
        self.assertNotIn("membre2 : ", second)  # la mention en cours reste décrite à part
        self.assertIn("Rika : Réponse un", second)

    async def test_late_heavy_split_next_group_sees_previous_answer(self):
        # En retard, le contexte s'arrête à la dernière mention, mais la 2e réponse voit quand même
        # la 1re mention et la réponse que le bot vient d'y faire.
        bot, mentions = self.heavy_split_setup(late=True)
        await ChatBot.answer_mention(bot, mentions, late=mock.Mock())
        first, second = (call.args[1][0][0] for call in bot.claude.ask.await_args_list)
        self.assertNotIn("Réponse un", first)
        self.assertIn("membre1 : @Rika lis ça", second)
        self.assertIn("Rika : Réponse un", second)
        self.assertNotIn("membre2 : ", second)
        self.assertNotIn("après", second)
        self.assertLess(second.index("membre1 : @Rika lis ça"), second.index("Rika : Réponse un"))

    async def test_all_refused_no_call(self):
        bot = fake_bot()
        channel = fake_channel([])
        bot.mention_part = mock.AsyncMock(return_value=None)
        await ChatBot.answer_mention(bot, [fake_mention(1, "regarde", channel)])
        bot.claude.ask.assert_not_awaited()
        bot.send_answer.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
