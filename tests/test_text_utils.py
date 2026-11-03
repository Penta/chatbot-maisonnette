"""Tests du découpage des messages et du retrait de la réflexion. Lancer depuis la racine du projet : python -m unittest"""

import unittest

from bot.text_utils import DISCORD_LIMIT, split_message, strip_reasoning

# Presque un message entier : la ligne suivante, même courte, passe au morceau d'après.
FULL = "a" * 1990 + "\n" + "b" * 5


def code_block(language: str, count: int) -> str:
    """Un bloc de code de `count` lignes numérotées, fermé."""
    lines = "\n".join(f"print('ligne {i} du bloc')" for i in range(count))
    return f"```{language}\n{lines}\n```"


class SplitMessageTest(unittest.TestCase):
    def assert_valid(self, chunks: list[str], limit: int = DISCORD_LIMIT) -> None:
        """Aucun morceau vide, aucun morceau au-delà de la limite, et chaque bloc de code refermé."""
        for chunk in chunks:
            self.assertTrue(chunk.strip(), chunks)
            self.assertLessEqual(len(chunk), limit, chunk)
            fences = [line for line in chunk.split("\n") if line.lstrip().startswith("```")]
            self.assertEqual(len(fences) % 2, 0, chunk)

    def test_short_text_unchanged(self):
        self.assertEqual(split_message("Bonjour !"), ["Bonjour !"])
        self.assertEqual(split_message("a" * DISCORD_LIMIT), ["a" * DISCORD_LIMIT])

    def test_surrounding_spaces_removed(self):
        self.assertEqual(split_message("  \n Bonjour !\n\n"), ["Bonjour !"])

    def test_empty_text(self):
        self.assertEqual(split_message(""), [])
        self.assertEqual(split_message(" \n\n "), [])

    def test_cut_between_lines(self):
        lines = [f"Ligne numéro {i}, avec un peu de texte pour la remplir." for i in range(100)]
        chunks = split_message("\n".join(lines))
        self.assertGreater(len(chunks), 1)
        self.assert_valid(chunks)
        # Aucune ligne n'est coupée en deux : on retrouve exactement les lignes de départ.
        self.assertEqual([line for chunk in chunks for line in chunk.split("\n")], lines)

    def test_code_block_closed_and_reopened(self):
        text = "Voici le code :\n" + code_block("python", 150) + "\nEt voilà."
        chunks = split_message(text)
        self.assertGreater(len(chunks), 1)
        self.assert_valid(chunks)
        self.assertTrue(chunks[0].endswith("\n```"))
        for chunk in chunks[1:]:
            self.assertTrue(chunk.startswith("```python\n"), chunk[:30])
        # Une fois les ``` ajoutés retirés, le code est entier et dans l'ordre.
        code = [line for chunk in chunks for line in chunk.split("\n") if line.startswith("print(")]
        self.assertEqual(code, [f"print('ligne {i} du bloc')" for i in range(150)])

    def test_code_block_without_language(self):
        chunks = split_message(code_block("", 150))
        self.assert_valid(chunks)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("```\nprint("), chunk[:30])

    def test_closed_block_not_closed_again(self):
        # Le bloc se ferme dans le dernier morceau : rien n'est ajouté après lui.
        text = FULL + "\n" + code_block("js", 3) + "\nFin du message."
        self.assertEqual(split_message(text), [FULL, code_block("js", 3) + "\nFin du message."])

    def test_text_ending_with_closed_block(self):
        text = FULL + "\n" + code_block("js", 3)
        self.assertEqual(split_message(text), [FULL, code_block("js", 3)])

    def test_unclosed_block_left_as_is(self):
        # Un bloc que le modèle n'a pas fermé ne l'est pas artificiellement dans le dernier morceau.
        text = FULL + "\n```python\nprint('oups')"
        self.assertEqual(split_message(text), [FULL, "```python\nprint('oups')"])

    def test_no_chunk_with_reopened_fence_only(self):
        # Le bloc se ferme juste au début d'un nouveau morceau : pas de morceau « ```python » seul.
        text = code_block("python", 150)
        chunks = split_message(text)
        self.assert_valid(chunks)
        self.assertNotIn("```python", chunks)

    def test_long_line_cut_at_space(self):
        words = [f"mot{i}" for i in range(1000)]
        chunks = split_message(" ".join(words))
        self.assertGreater(len(chunks), 1)
        self.assert_valid(chunks)
        # Aucun mot n'est coupé en deux.
        self.assertEqual([word for chunk in chunks for word in chunk.split()], words)

    def test_long_line_without_space_cut_clean(self):
        text = "x" * 5000
        chunks = split_message(text)
        self.assert_valid(chunks)
        self.assertEqual("".join(chunks), text)

    def test_space_too_early_ignored(self):
        # Le seul espace est tout au début : couper là ferait un morceau minuscule, la coupe est nette.
        text = "ab " + "x" * 3000
        chunks = split_message(text)
        self.assert_valid(chunks)
        self.assertGreater(len(chunks[0]), DISCORD_LIMIT // 2)
        self.assertEqual("".join(chunks).replace(" ", ""), text.replace(" ", ""))

    def test_long_line_inside_code_block(self):
        text = "```python\n" + "y" * 4500 + "\n```"
        chunks = split_message(text)
        self.assert_valid(chunks)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("```python\n"), chunk[:30])
            self.assertTrue(chunk.endswith("\n```"), chunk[-30:])
        self.assertEqual("".join(c.removeprefix("```python\n").removesuffix("\n```") for c in chunks), "y" * 4500)

    def test_small_limit(self):
        text = ("Un peu de texte avant le code.\n" + code_block("python", 12) + "\n"
                + "un mot " * 30 + "\n" + "z" * 120 + "\nFin.")
        for limit in (40, 60, 100):
            with self.subTest(limit=limit):
                chunks = split_message(text, limit)
                self.assertGreater(len(chunks), 1)
                self.assert_valid(chunks, limit)

    def test_opening_fence_at_end_of_chunk(self):
        # La ligne d'ouverture d'un bloc (```py) qui remplit presque le morceau passe au morceau suivant,
        # au lieu de laisser un bloc vide « ```py\n``` » qui pourrait dépasser la limite.
        for before in range(1990, 2000):
            with self.subTest(before=before):
                text = "a" * before + "\n```py\n" + "b" * 10 + "\nfin\n```"
                chunks = split_message(text)
                self.assert_valid(chunks)
                self.assertFalse(chunks[0].endswith("```py\n```"), chunks[0][-20:])

    def test_closing_fence_at_end_of_chunk(self):
        # La ligne qui ferme un bloc (```) entre dans un morceau presque plein : elle n'a pas à garder
        # la place d'une fermeture artificielle, sinon le morceau suivant commence par un bloc vide.
        for last in range(980, 992):
            with self.subTest(last=last):
                text = "```py\n" + "a" * 1000 + "\n" + "b" * last + "\n```\nSuite du message."
                chunks = split_message(text)
                self.assert_valid(chunks)
                self.assertFalse(any(chunk.startswith("```py\n```") for chunk in chunks), chunks[-1][:20])

    def test_long_closing_fence_line(self):
        # Une ligne de fermeture trop longue est découpée : le bloc reste ouvert entre ses morceaux,
        # qui doivent donc garder la place de le refermer.
        for after in range(20, 60):
            with self.subTest(after=after):
                text = "```py\n" + "a" * 5 + "\n```" + "x" * after + " fin\nSuite."
                for chunk in split_message(text, 40):
                    self.assertLessEqual(len(chunk), 40, chunk)


class StripReasoningTest(unittest.TestCase):
    def test_think_block_at_start_removed(self):
        self.assertEqual(strip_reasoning("<think>Il faut être poli.</think>\n\nBonjour !"), "Bonjour !")

    def test_other_tags_removed(self):
        for tag in ("think", "thinking", "reasoning", "function_calls", "invoke"):
            with self.subTest(tag=tag):
                self.assertEqual(strip_reasoning(f"<{tag}>réflexion</{tag}>Réponse"), "Réponse")

    def test_tag_with_attributes_and_case(self):
        text = '<INVOKE name="web_search">\n<parameter>météo</parameter>\n</invoke> Il fait beau.'
        self.assertEqual(strip_reasoning(text), "Il fait beau.")

    def test_multiline_block_and_leading_spaces(self):
        self.assertEqual(strip_reasoning("\n  <thinking>\nune\ndeux\n</thinking>\nSalut"), "Salut")

    def test_several_blocks_at_start(self):
        text = "<think>un</think>\n<function_calls>appel</function_calls>\nRéponse"
        self.assertEqual(strip_reasoning(text), "Réponse")

    def test_tag_later_in_text_kept(self):
        text = "Les modèles écrivent parfois <think>ceci</think> avant de répondre."
        self.assertEqual(strip_reasoning(text), text)

    def test_tag_in_code_kept(self):
        text = "Exemple :\n```xml\n<thinking>texte</thinking>\n```"
        self.assertEqual(strip_reasoning(text), text)

    def test_unclosed_block_kept(self):
        # Sans balise de fin, rien n'est retiré : on ne sait pas où la réflexion s'arrête.
        self.assertEqual(strip_reasoning("<think>réflexion sans fin"), "<think>réflexion sans fin")

    def test_mismatched_tags_kept(self):
        self.assertEqual(strip_reasoning("<think>a</thinking> b"), "<think>a</thinking> b")

    def test_only_reasoning(self):
        self.assertEqual(strip_reasoning("<think>rien d'autre</think>  "), "")

    def test_plain_text_stripped(self):
        self.assertEqual(strip_reasoning("  Bonjour !\n"), "Bonjour !")


if __name__ == "__main__":
    unittest.main()
