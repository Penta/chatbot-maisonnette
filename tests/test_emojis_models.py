"""Tests des emojis du serveur et des noms de modèles. Lancer depuis la racine du projet : python -m unittest"""

import unittest
from types import SimpleNamespace

from bot.emojis import discord_emojis, emoji_names, short_emojis, usable_emojis
from bot.models import display_name, is_known, thinking_off


class FakeEmoji:
    """Emoji personnalisé, écrit comme discord.Emoji (<:nom:id>, <a:nom:id> s'il est animé)."""

    def __init__(self, name: str, id: int, animated: bool = False, available: bool = True, roles=()):
        self.name, self.id, self.animated, self.available, self.roles = name, id, animated, available, list(roles)

    def __str__(self) -> str:
        return f"<{'a' if self.animated else ''}:{self.name}:{self.id}>"


def guild(*emojis: FakeEmoji, roles=()) -> SimpleNamespace:
    return SimpleNamespace(emojis=list(emojis), me=SimpleNamespace(roles=list(roles)))


HAP = FakeEmoji("hap", 111)
DANCE = FakeEmoji("dance", 222, animated=True)
GUILD = guild(HAP, DANCE)


class DiscordEmojisTest(unittest.TestCase):
    def test_name_replaced(self):
        self.assertEqual(discord_emojis("Coucou :hap: !", GUILD), "Coucou <:hap:111> !")

    def test_animated(self):
        self.assertEqual(discord_emojis(":dance:", GUILD), "<a:dance:222>")

    def test_several_in_a_row(self):
        self.assertEqual(discord_emojis(":hap::dance: :hap:", GUILD), "<:hap:111><a:dance:222> <:hap:111>")

    def test_case_tolerated(self):
        self.assertEqual(discord_emojis(":Hap: :DANCE:", GUILD), "<:hap:111> <a:dance:222>")

    def test_unknown_name_kept(self):
        self.assertEqual(discord_emojis(":inconnu: à 12:30:45", GUILD), ":inconnu: à 12:30:45")

    def test_code_kept(self):
        for text in ("```\n:hap:\n```", "```py\nprint(':hap:')\n```", "`:hap:`"):
            with self.subTest(text):
                self.assertEqual(discord_emojis(text, GUILD), text)
        self.assertEqual(discord_emojis("`:hap:` donne :hap:", GUILD), "`:hap:` donne <:hap:111>")

    def test_already_a_code(self):
        self.assertEqual(discord_emojis("<:hap:111> <a:dance:222>", GUILD), "<:hap:111> <a:dance:222>")

    def test_no_guild(self):
        self.assertEqual(discord_emojis("Coucou :hap: !", None), "Coucou :hap: !")
        self.assertEqual(discord_emojis("Coucou :hap: !", guild()), "Coucou :hap: !")

    def test_unusable_emoji_not_replaced(self):
        hidden = guild(FakeEmoji("hap", 111, available=False), FakeEmoji("vip", 333, roles=["role_vip"]))
        self.assertEqual(discord_emojis(":hap: :vip:", hidden), ":hap: :vip:")


class ShortEmojisTest(unittest.TestCase):
    def test_codes_replaced_by_names(self):
        self.assertEqual(short_emojis("Coucou <:hap:111> <a:dance:222> !"), "Coucou :hap: :dance: !")

    def test_text_without_emoji_kept(self):
        self.assertEqual(short_emojis("Coucou :hap: <@123> <#456>"), "Coucou :hap: <@123> <#456>")

    def test_round_trip(self):
        text = "Coucou :hap: et :dance: !"
        self.assertEqual(short_emojis(discord_emojis(text, GUILD)), text)


class UsableEmojisTest(unittest.TestCase):
    def test_sorted_by_name(self):
        zebra, apple, upper = FakeEmoji("zebre", 1), FakeEmoji("abricot", 2), FakeEmoji("Banane", 3)
        self.assertEqual(usable_emojis(guild(zebra, apple, upper)), [apple, upper, zebra])
        self.assertEqual(emoji_names(guild(zebra, apple, upper)), ["abricot", "Banane", "zebre"])

    def test_unavailable_excluded(self):
        self.assertEqual(emoji_names(guild(HAP, FakeEmoji("parti", 4, available=False))), ["hap"])

    def test_roles(self):
        vip = FakeEmoji("vip", 5, roles=["role_vip"])
        self.assertEqual(emoji_names(guild(HAP, vip)), ["hap"])
        self.assertEqual(emoji_names(guild(HAP, vip, roles=["role_autre"])), ["hap"])
        self.assertEqual(emoji_names(guild(HAP, vip, roles=["role_autre", "role_vip"])), ["hap", "vip"])

    def test_bot_not_a_member(self):
        vip = FakeEmoji("vip", 5, roles=["role_vip"])
        self.assertEqual(emoji_names(SimpleNamespace(emojis=[HAP, vip], me=None)), ["hap"])

    def test_no_guild(self):
        self.assertEqual(usable_emojis(None), [])
        self.assertEqual(emoji_names(None), [])


class ModelsTest(unittest.TestCase):
    def test_is_known(self):
        for model in ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"):
            with self.subTest(model):
                self.assertTrue(is_known(model))
        for model in ("claude-sonnet-5-6", "claude-sonet-5-5", "gpt-4", ""):
            with self.subTest(model):
                self.assertFalse(is_known(model))

    def test_display_name(self):
        names = {
            "claude-sonnet-5-5": "Sonnet 5.5",
            "claude-opus-5-5": "Opus 5.5",
            "claude-sonnet-4-5-20250929": "Sonnet 4.5",
            "claude-opus-5": "Opus 5",
            "claude-opus-4-20250514": "Opus 4",
            "claude-haiku-4-10": "Haiku 4.10",
        }
        for model, name in names.items():
            with self.subTest(model):
                self.assertEqual(display_name(model), name)

    def test_display_name_unusual_id_kept(self):
        for model in ("claude-3-5-sonnet-20241022", "claude-sonnet-5-5-latest", "Claude-Sonnet-5-5",
                      "claude-sonnet", "gpt-4", ""):
            with self.subTest(model):
                self.assertEqual(display_name(model), model)

    def test_thinking_off(self):
        self.assertEqual(thinking_off("claude-sonnet-5-5"), {"type": "between_tools"})
        self.assertIsNone(thinking_off("claude-opus-5-5"))
        self.assertEqual(thinking_off("claude-haiku-5-5"), {"type": "disabled"})
        self.assertEqual(thinking_off("claude-opus-4-8"), {"type": "disabled"})  # modèle plus ancien
        self.assertEqual(thinking_off("modele-inconnu"), {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
