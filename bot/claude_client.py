"""Communication avec l'API de Claude (Anthropic)."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .config import Settings
from .models import thinking_off

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

# Ajouté quand WEB_SEARCH=on.
WEB_NOTE = (
    "\n\nTu as accès à Internet (recherche web et lecture de pages). "
    "Ne t'en sers que si la question porte sur l'actualité, des informations récentes ou précises "
    "que tu ne connais pas, ou un lien partagé dans la conversation ; sinon réponds directement. "
    "Garde des réponses courtes ; quand c'est utile, nomme simplement ta source dans ta phrase "
    "(par exemple « selon Famitsu »), sans recopier d'adresse. "
    "Ne commente pas tes recherches (pas de « je vais vérifier », « je regarde le résultat »…) : donne directement la réponse. "
    "Dans les messages précédents, la mention « [recherche web : …] » signifie que tu as bien cherché sur Internet "
    "pour y répondre : ta réponse d'alors s'appuyait sur de vrais résultats, même si tu ne les vois plus. "
    "Ne prétends donc jamais avoir inventé une réponse marquée ainsi."
)

# Nombre maximal de relances quand l'API met la réponse en pause pendant les recherches.
MAX_CONTINUATIONS = 3


class ClaudeError(Exception):
    """Erreur lors de l'appel à Claude, avec un message affichable sur Discord."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class ImageData:
    media_type: str
    data: bytes


@dataclass
class Answer:
    text: str
    web_used: bool  # Claude a fait au moins une recherche web ou lu une page


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

    async def ask(self, history: list[dict], user_text: str, image: ImageData | None = None) -> Answer:
        """Envoie l'historique + le nouveau message, et renvoie la réponse de Claude."""
        messages = list(history) + [{"role": "user", "content": self.build_user_content(user_text, image)}]

        system = self.personality() + MEMORY_NOTE
        if self.settings.web_search:
            system += WEB_NOTE

        # Cache de prompt : les outils et la personnalité, identiques d'un message à l'autre, sont
        # relus depuis le cache à 10 % du prix (pendant 5 minutes, prolongées à chaque message).
        # Le marqueur sur "system" couvre ce bloc fixe ; celui de premier niveau couvre en plus
        # l'historique, qui change à chaque message mais reprend le début de la requête précédente.
        params: dict = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
        }
        # Par défaut, les modèles récents réfléchissent avant de répondre, et cette réflexion
        # consomme le budget MAX_TOKENS. Avec THINKING=off, on la coupe de la façon propre au
        # modèle (voir models.py) ; sur Opus 5.5, impossible : seul EFFORT la réduit.
        if self.settings.thinking:
            params["thinking"] = {"type": "adaptive"}
        elif (off := thinking_off(self.settings.model)) is not None:
            params["thinking"] = off
        if self.settings.effort:
            params["extra_body"] = {"output_config": {"effort": self.settings.effort}}
        if self.settings.web_search:
            # Outils exécutés par Anthropic : pas de boucle d'outils à gérer de notre côté.
            params["tools"] = [
                {"type": "web_search_20260209", "name": "web_search", "max_uses": self.settings.web_max_uses},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": self.settings.web_max_uses},
            ]

        responses = [await self._call(params, image)]
        # Lors de longues recherches, l'API peut rendre la main avant la fin (pause_turn) :
        # on renvoie alors sa réponse partielle telle quelle pour qu'elle continue.
        while responses[-1].stop_reason == "pause_turn" and len(responses) <= MAX_CONTINUATIONS:
            messages = messages + [{"role": "assistant", "content": responses[-1].content}]
            responses.append(await self._call({**params, "messages": messages}, image))

        text = self._final_text(responses)
        web_used = self._log_usage(responses)

        stop_reason = responses[-1].stop_reason
        if stop_reason == "max_tokens":
            logger.warning("Réponse coupée par MAX_TOKENS (%s) : augmente MAX_TOKENS dans le .env "
                           "si cela arrive souvent.", self.settings.max_tokens)
            text += " […]"
        elif stop_reason == "pause_turn":
            logger.warning("Recherche web toujours en cours après %s relances : réponse partielle.", MAX_CONTINUATIONS)
        if not text:
            logger.warning("Réponse vide de Claude (stop_reason=%s).", stop_reason)
            raise ClaudeError("Je n'ai rien trouvé à répondre à ça.")
        return Answer(text, web_used)

    async def _call(self, params: dict, image: ImageData | None):
        try:
            return await self._client.messages.create(**params)
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

    @staticmethod
    def _final_text(responses) -> str:
        """Texte de la réponse finale, sans les commentaires écrits entre deux recherches."""
        parts: list[str] = []
        for response in responses:
            for block in response.content:
                if block.type == "text":
                    parts.append(block.text)
                elif block.type not in ("thinking", "redacted_thinking"):
                    # Appel ou résultat d'outil : le texte écrit avant n'était qu'un commentaire.
                    parts = []
        return "".join(parts).strip()

    def _log_usage(self, responses) -> bool:
        """Journalise la consommation et indique si Claude a utilisé Internet."""
        input_tokens = sum(r.usage.input_tokens for r in responses)
        cache_write = sum(r.usage.cache_creation_input_tokens or 0 for r in responses)
        cache_read = sum(r.usage.cache_read_input_tokens or 0 for r in responses)
        output_tokens = sum(r.usage.output_tokens for r in responses)
        searches = fetches = 0
        for r in responses:
            if r.usage.server_tool_use:
                searches += r.usage.server_tool_use.web_search_requests or 0
                fetches += r.usage.server_tool_use.web_fetch_requests or 0
        price_input, price_output = self.settings.price_input_per_mtok, self.settings.price_output_per_mtok
        if price_input is None or price_output is None:
            cost = "inconnu (renseigne PRICE_INPUT_PER_MTOK et PRICE_OUTPUT_PER_MTOK dans le .env)"
        else:
            # Écriture en cache : 125 % du prix d'entrée ; lecture depuis le cache : 10 %.
            amount = ((input_tokens + cache_write * 1.25 + cache_read * 0.1) * price_input
                      + output_tokens * price_output) / 1_000_000
            amount += searches * self.settings.price_web_search_per_1000 / 1000
            cost = f"{amount:.5f} $"
        web = f", recherches web : {searches}, pages lues : {fetches}" if searches or fetches else ""
        logger.info(
            "Appel Claude (%s) — entrée : %s tokens (+ cache : %s lus, %s écrits), sortie : %s tokens%s, "
            "coût estimé : %s",
            responses[-1].model, input_tokens, cache_read, cache_write, output_tokens, web, cost,
        )
        return bool(searches or fetches)
