"""Communication avec l'API de Claude (Anthropic)."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .config import Settings

logger = logging.getLogger("chatbot.claude")

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

DEFAULT_PERSONALITY = (
    "Tu es un membre sympathique d'un serveur Discord francophone. "
    "Tu réponds toujours en français, avec un ton naturel, amical et décontracté, "
    "et des réponses plutôt courtes, adaptées à une discussion de chat. "
    "Chaque message que tu reçois indique la date, l'heure et le pseudo de son auteur ; "
    "plusieurs personnes peuvent parler dans la même conversation. "
    "Quand on te montre une image, décris ce que tu vois puis réponds à la question s'il y en a une."
)


# Ajouté automatiquement à la fin de la personnalité, quel que soit le contenu du fichier.
MEMORY_NOTE = (
    "\n\nNote technique sur ta mémoire : les images ne sont pas conservées d'un message à l'autre. "
    "Dans les messages précédents, la mention « [image jointe : …] » signifie qu'une image a bien été envoyée "
    "et que tu l'as réellement vue et analysée à ce moment-là ; ta réponse d'alors en décrit le contenu. "
    "Ne dis donc jamais que tu ne l'as pas vue, et ne t'excuse pas à ce sujet. "
    "Si on te demande un détail de cette image que ta réponse précédente ne mentionnait pas, "
    "explique simplement que tu ne gardes pas les images en mémoire et propose qu'on te la renvoie."
)


class ClaudeError(Exception):
    """Erreur lors de l'appel à Claude, avec un message affichable sur Discord."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class ImageData:
    media_type: str
    data: bytes


class ClaudeClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Le SDK réessaie tout seul les erreurs temporaires (surcharge, limite de requêtes…).
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.api_timeout,
            max_retries=3,
        )
        self._personality_cache: tuple[float, str] | None = None

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------ personnalité

    def personality(self) -> str:
        """Lit personality_prompt.txt (relu automatiquement s'il a été modifié)."""
        path: Path = self.settings.personality_file
        try:
            mtime = path.stat().st_mtime
            if self._personality_cache is None or self._personality_cache[0] != mtime:
                text = path.read_text(encoding="utf-8").strip()
                self._personality_cache = (mtime, text or DEFAULT_PERSONALITY)
                logger.info("Personnalité chargée depuis %s", path)
            return self._personality_cache[1]
        except FileNotFoundError:
            if self._personality_cache != (0.0, DEFAULT_PERSONALITY):
                logger.warning("%s introuvable : utilisation de la personnalité par défaut.", path)
                self._personality_cache = (0.0, DEFAULT_PERSONALITY)
            return DEFAULT_PERSONALITY

    # ------------------------------------------------------------------ appel

    @staticmethod
    def build_user_content(text: str, image: ImageData | None = None) -> str | list[dict]:
        if image is None:
            return text
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": base64.standard_b64encode(image.data).decode("ascii"),
                },
            },
            {"type": "text", "text": text},
        ]

    async def ask(self, history: list[dict], user_text: str, image: ImageData | None = None) -> str:
        """Envoie l'historique + le nouveau message, et renvoie la réponse de Claude."""
        messages = list(history) + [{"role": "user", "content": self.build_user_content(user_text, image)}]

        params: dict = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.personality() + MEMORY_NOTE,
            "messages": messages,
        }
        # Par défaut, Sonnet 5.5 réfléchit avant de répondre, et cette réflexion consomme
        # le budget MAX_TOKENS. "between_tools" désactive la réflexion préalable
        # (le bot n'utilisant pas d'outils, il n'y a alors aucune réflexion).
        params["thinking"] = {"type": "adaptive"} if self.settings.thinking else {"type": "between_tools"}
        if self.settings.effort:
            params["extra_body"] = {"output_config": {"effort": self.settings.effort}}

        try:
            response = await self._client.messages.create(**params)
        except anthropic.RateLimitError:
            logger.warning("Limite de requêtes Anthropic atteinte.")
            raise ClaudeError("Je reçois trop de messages en ce moment, réessaie dans une minute.")
        except anthropic.APITimeoutError:
            logger.warning("Délai dépassé lors de l'appel à Claude.")
            raise ClaudeError("Je mets trop de temps à répondre, réessaie dans un instant.")
        except anthropic.APIConnectionError as exc:
            logger.error("Connexion à l'API impossible : %s", exc)
            raise ClaudeError("Je n'arrive pas à joindre mon cerveau pour l'instant, réessaie plus tard.")
        except anthropic.BadRequestError as exc:
            logger.error("Requête refusée par l'API : %s", exc)
            if image is not None:
                raise ClaudeError("Je n'ai pas réussi à lire cette image, essaie avec une autre.")
            raise ClaudeError("Ta demande a été refusée par l'API, essaie de la reformuler.")
        except anthropic.AuthenticationError:
            logger.critical("Clé API Anthropic invalide : vérifie ANTHROPIC_API_KEY dans le .env.")
            raise ClaudeError("Je suis mal configuré, préviens l'administrateur du bot.")
        except anthropic.APIStatusError as exc:
            logger.error("Erreur de l'API (%s) : %s", exc.status_code, exc)
            raise ClaudeError("Une erreur est survenue de mon côté, réessaie dans un instant.")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        self._log_usage(response)

        if response.stop_reason == "max_tokens":
            logger.warning("Réponse coupée par MAX_TOKENS (%s) : augmente MAX_TOKENS dans le .env "
                           "si cela arrive souvent.", self.settings.max_tokens)
            text += " […]"
        if not text:
            logger.warning("Réponse vide de Claude (stop_reason=%s).", response.stop_reason)
            raise ClaudeError("Je n'ai rien trouvé à répondre à ça.")
        return text

    def _log_usage(self, response) -> None:
        usage = response.usage
        cost = (usage.input_tokens * self.settings.price_input_per_mtok
                + usage.output_tokens * self.settings.price_output_per_mtok) / 1_000_000
        logger.info(
            "Appel Claude (%s) — entrée : %s tokens, sortie : %s tokens, coût estimé : %.5f $",
            response.model, usage.input_tokens, usage.output_tokens, cost,
        )
