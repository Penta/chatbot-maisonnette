"""Tests des longs blocs de code envoyés en fichier joint et résumés dans l'historique.
Lancer depuis la racine du projet : python -m unittest"""

import asyncio
import contextlib
import unittest
from datetime import datetime, timezone
from unittest import mock

from bot.code_files import MIN_LINES, outgoing, prepare
from bot.discord_bot import ChatBot, PendingMessage


def code(lines: int, start: str = "<p>ligne") -> str:
    return "\n".join(f"{start} {n}</p>" for n in range(1, lines + 1))


PAGE = code(40)
ANSWER = f"Tiens, ta page d'accueil ! J'ai mis le menu en haut 😊\n\n```html index.html\n{PAGE}\n```\n\nDis-moi si ça te va."


class PrepareTest(unittest.TestCase):
    def test_long_block_becomes_a_file(self):
        # Penta demande une page web : Rika répond avec 40 lignes de HTML. Le message garde ses mots à elle,
        # le code part dans index.html, et la mémoire ne garde qu'un rappel du fichier.
        prepared = prepare(ANSWER)
        self.assertEqual(prepared.text, "Tiens, ta page d'accueil ! J'ai mis le menu en haut 😊\n\nDis-moi si ça te va.")
        self.assertEqual([(f.name, f.content) for f in prepared.files], [("index.html", PAGE + "\n")])
        self.assertEqual(prepared.memory, "Tiens, ta page d'accueil ! J'ai mis le menu en haut 😊\n\n"
                                          "[code envoyé en fichier joint : index.html, 40 lignes]\n\nDis-moi si ça te va.")
        self.assertNotIn("<p>", prepared.memory)

    def test_short_block_stays_in_the_message(self):
        # Un extrait de quelques lignes se lit directement dans Discord, sans ouvrir de fichier.
        answer = f"Ajoute juste ça :\n```css\n{code(3, 'a')}\n```"
        prepared = prepare(answer)
        self.assertEqual((prepared.text, prepared.files, prepared.memory), (answer, [], answer))

    def test_long_single_line_becomes_a_file(self):
        answer = "Le style :\n```css\n" + "a{color:red}" * 200 + "\n```"
        self.assertEqual([f.name for f in prepare(answer).files], ["code.css"])

    def test_file_names(self):
        # Nom donné par Rika, sinon un nom d'après le langage ; deux fois le même nom : le second est numéroté.
        for opening, name in (("```html index.html", "index.html"), ("```style.css", "style.css"),
                              ("```javascript", "code.js"), ("```", "code.txt"), ("```klingon", "code.txt"),
                              ("```html ../../secret.html", "code.html")):
            with self.subTest(opening=opening):
                self.assertEqual(prepare(f"{opening}\n{PAGE}\n```").files[0].name, name)
        twice = prepare(f"```js\n{PAGE}\n```\n```js\n{PAGE}\n```")
        self.assertEqual([f.name for f in twice.files], ["code.js", "code-2.js"])

    def test_code_only_answer(self):
        prepared = prepare(f"```html index.html\n{PAGE}\n```")
        self.assertEqual(prepared.text, "")
        self.assertEqual(prepared.memory, "[code envoyé en fichier joint : index.html, 40 lignes]")

    def test_answer_cut_in_the_middle_of_the_code(self):
        # Réponse coupée par MAX_TOKENS au milieu du code : le fichier est envoyé tel quel, sans le « […] »
        # ajouté à la coupure, qui reste à la fin du message ; la mémoire sait que le fichier est incomplet.
        prepared = prepare(f"Voilà le site :\n```html index.html\n{PAGE} […]")
        self.assertEqual(prepared.text, "Voilà le site : […]")
        self.assertEqual(prepared.files[0].content, PAGE + "\n")
        self.assertEqual(prepared.memory, "Voilà le site :\n"
                                          "[code envoyé en fichier joint : index.html, 40 lignes, coupé avant la fin]")

    def test_threshold(self):
        self.assertEqual(prepare(f"```\n{code(MIN_LINES - 1)}\n```").files, [])
        self.assertEqual(len(prepare(f"```\n{code(MIN_LINES)}\n```").files), 1)


class OutgoingTest(unittest.TestCase):
    def test_files_go_with_the_last_message(self):
        prepared = prepare(ANSWER)
        messages = outgoing("x" * 3000, prepared.files)
        self.assertEqual([extra for _, extra in messages][:-1], [{}])
        self.assertEqual([f.filename for f in messages[-1][1]["files"]], ["index.html"])

    def test_files_alone(self):
        messages = outgoing("", prepare(f"```html index.html\n{PAGE}\n```").files)
        self.assertEqual(len(messages), 1)
        self.assertIsNone(messages[0][0])

    def test_no_file(self):
        self.assertEqual(outgoing("Coucou", []), [("Coucou", {})])


@mock.patch("bot.discord_bot.typing", lambda channel: contextlib.nullcontext())
@mock.patch("bot.discord_bot.emoji_names", lambda guild: [])
class ChannelTest(unittest.IsolatedAsyncioTestCase):
    async def test_code_sent_as_file_and_summarized_in_history(self):
        # Dans le salon dédié : un seul message Discord avec le fichier, au lieu de plusieurs messages de code,
        # et l'historique ne garde pas le code (il serait relu à chaque message suivant).
        bot = mock.Mock()
        bot.settings.reply_pause = 0
        bot._pending, bot._preparing, bot._waiters, bot._mentions = {}, {}, {}, {}
        bot._deleted, bot._waiting = set(), set()
        bot.available.return_value = True
        bot.history.lock.return_value = asyncio.Lock()
        bot.history.get.return_value = []
        bot.history.add_exchange = mock.AsyncMock()
        bot.presence_note.return_value = None
        bot.member_notes = mock.AsyncMock(return_value="")
        bot.claude.ask = mock.AsyncMock(return_value=mock.Mock(text=ANSWER, web_used=False))
        bot.send_answer = ChatBot.send_answer.__get__(bot)
        bot.answer_main_channel = lambda channel, late=None: ChatBot.answer_main_channel(bot, channel, late)
        channel = mock.Mock(id=10)
        channel.send = mock.AsyncMock()
        message = mock.Mock(id=1, channel=channel, guild=None, attachments=[], content="fais-moi un site",
                            stickers=[], created_at=datetime(2026, 11, 7, tzinfo=timezone.utc))
        line = {"id": 1, "date": "07/11/2026 18:00", "author": "Penta", "content": "fais-moi un site"}

        async def read(msg, content):
            return PendingMessage(msg, line, "", [])
        bot.read_main_message = read

        await ChatBot.handle_main_channel(bot, message)

        channel.send.assert_awaited_once()
        sent = channel.send.await_args
        self.assertEqual(sent.args[0], "Tiens, ta page d'accueil ! J'ai mis le menu en haut 😊\n\nDis-moi si ça te va.")
        self.assertEqual([f.filename for f in sent.kwargs["files"]], ["index.html"])
        remembered = bot.history.add_exchange.await_args.args[2]
        self.assertIn("[code envoyé en fichier joint : index.html, 40 lignes]", remembered)
        self.assertNotIn("<p>", remembered)


if __name__ == "__main__":
    unittest.main()
