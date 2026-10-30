"""Communication avec l'API de Claude (Anthropic)."""

from __future__ import annotations

import base64
import logging
import re
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

# Ajouté lui aussi automatiquement : les messages reçus pendant une réponse sont regroupés.
GROUP_NOTE = (
    "\n\nUn même message peut regrouper plusieurs messages Discord, un par ligne, envoyés "
    "pendant que tu répondais au précédent : réponds à l'ensemble en une seule réponse, "
    "naturellement, sans signaler qu'ils ont été regroupés."
)

# Ajouté lui aussi automatiquement. Formulation générique recommandée par Anthropic : nommer les
# balises de réflexion, ou interdire de réfléchir, rend au contraire leur apparition plus probable.
OUTPUT_NOTE = "\n\nN'inclus pas de balises XML internes ou système dans ta réponse."

# Ajouté quand WEB_SEARCH=on.
WEB_NOTE = (
    "\n\nTu as accès à Internet (recherche web et lecture de pages). "
    "Ne t'en sers que si la question porte sur l'actualité, des informations récentes ou précises "
    "que tu ne connais pas, ou un lien partagé dans la conversation ; sinon réponds directement. "
    "Garde des réponses courtes ; quand c'est utile, nomme simplement ta source dans ta phrase "
    "(par exemple « selon Famitsu »), sans recopier d'adresse. "
    # Le texte écrit avant un appel d'outil n'est jamais affiché (voir _final_text). L'interdire
    # pousse au contraire le modèle à écrire ses appels d'outils sous forme de texte.
    "Quand tu utilises un outil, tu peux d'abord dire une courte phrase. "
    "Si aucun outil ne permet de faire ce qu'on te demande, dis-le au lieu de deviner. "
    "Dans les messages précédents, la mention « [recherche web : …] » signifie que tu as bien cherché sur Internet "
    "pour y répondre : ta réponse d'alors s'appuyait sur de vrais résultats, même si tu ne les vois plus. "
    "Ne prétends donc jamais avoir inventé une réponse marquée ainsi."
)

# Ajouté quand PRESENCE=on : le statut Discord en cours est indiqué au début des messages.
PRESENCE_NOTE = (
    "\n\nAu début d'un message, « [ton statut Discord : …] » indique ce que tu fais à ce moment-là "
    "(anime que tu regardes, musique que tu écoutes, jeu auquel tu joues, statut personnalisé ou disponibilité) ; "
    "tout le serveur le voit sous ton pseudo. C'est vraiment ce que tu es en train de faire : tu peux en parler "
    "naturellement si c'est pertinent ou si on te le demande, mais pas à chaque message, et sans recopier cette mention. "
    "Seul le statut du dernier message est d'actualité. "
    "Quand tu es absente, tu ne réponds qu'à ton retour, et en « Ne pas déranger », seulement au bout de "
    "quelques minutes : « [Ces messages sont arrivés pendant que tu étais …] » te l'indique. Réponds alors "
    "comme quelqu'un qui revient, naturellement, sans forcément t'excuser."
)

# Demandes de statut personnalisé (pensée du moment, raison d'une absence), écrit avec la
# personnalité du bot. Les clés sont les noms des statuts de presence.py.
CUSTOM_STATUS_REQUESTS = {
    "pensee": "Écris le statut personnalisé que tu affiches en ce moment sous ton pseudo Discord : "
              "une courte pensée du moment, dans ton style",
    "absent": "Tu t'absentes de ton PC pendant {duration}. Écris le statut personnalisé que tu affiches "
              "sous ton pseudo Discord pendant ce temps : une raison plausible, dans ton style",
    "ne_pas_deranger": "Tu passes en « Ne pas déranger » sur Discord pendant {duration}. Écris le statut "
                       "personnalisé que tu affiches sous ton pseudo pour expliquer pourquoi : une raison "
                       "plausible, dans ton style",
}
CUSTOM_STATUS_PROMPT = (
    "Nous sommes {now}. {request}, de 60 caractères au plus. Les emojis du serveur ne s'affichent pas "
    "dans un statut : si tu veux un emoji, prends un emoji classique (Unicode).{recent} "
    "Réponds uniquement par ce statut, sans guillemets ni explication."
)

CUSTOM_EMOJI = re.compile(r"<a?:\w+:\d+>")

# Erreur due à la configuration (clé, crédit, modèle, réglages) : seul l'administrateur peut la corriger.
CONFIG_ERROR = "Je suis mal configuré, préviens l'administrateur du bot."

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
    def build_user_content(parts: list[tuple[str, ImageData | None]]) -> str | list[dict]:
        """Un ou plusieurs messages à la suite, chacun précédé de son image éventuelle."""
        if all(image is None for _, image in parts):
            return "\n".join(text for text, _ in parts)
        content: list[dict] = []
        for text, image in parts:
            if image is not None:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": base64.standard_b64encode(image.data).decode("ascii"),
                    },
                })
            content.append({"type": "text", "text": text})
        return content

    async def ask(self, history: list[dict], parts: list[tuple[str, ImageData | None]],
                  cache_history: bool = True) -> Answer:
        """Envoie l'historique + le ou les nouveaux messages (texte, image), et renvoie la réponse de Claude.

        cache_history=False évite d'écrire les messages en cache quand l'appel suivant ne les
        reprendra pas (mentions : contexte différent à chaque fois)."""
        messages = list(history) + [{"role": "user", "content": self.build_user_content(parts)}]
        has_image = any(image is not None for _, image in parts)

        system = self.personality() + MEMORY_NOTE + GROUP_NOTE + OUTPUT_NOTE
        if self.settings.web_search:
            system += WEB_NOTE
        if self.settings.presence:
            system += PRESENCE_NOTE

        # Cache de prompt : les outils et la personnalité, identiques d'un message à l'autre, sont
        # relus depuis le cache à 10 % du prix (pendant 5 minutes, prolongées à chaque message).
        # Le marqueur sur "system" couvre ce bloc fixe ; celui de premier niveau couvre en plus
        # l'historique, qui change à chaque message mais reprend le début de la requête précédente.
        # Une écriture en cache coûte 25 % de plus : inutile si elle ne sera jamais relue.
        params: dict = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if cache_history:
            params["cache_control"] = {"type": "ephemeral"}
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

        responses = [await self._call(params, has_image)]
        # Lors de longues recherches, l'API peut rendre la main avant la fin (pause_turn) :
        # on renvoie alors sa réponse partielle telle quelle pour qu'elle continue.
        while responses[-1].stop_reason == "pause_turn" and len(responses) <= MAX_CONTINUATIONS:
            messages = messages + [{"role": "assistant", "content": responses[-1].content}]
            responses.append(await self._call({**params, "messages": messages}, has_image))

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

    async def write_custom_status(self, kind: str, now: str, duration: str, recent: list[str],
                                  max_length: int) -> str | None:
        """Statut personnalisé du bot (voir CUSTOM_STATUS_REQUESTS), ou None en cas d'échec."""
        recent_part = ""
        if recent:
            recent_part = " Ne reprends pas tes statuts récents : " + ", ".join(f"« {r} »" for r in recent) + "."
        prompt = CUSTOM_STATUS_PROMPT.format(
            now=now, request=CUSTOM_STATUS_REQUESTS[kind].format(duration=duration), recent=recent_part,
        )
        params: dict = {
            "model": self.settings.model,
            "max_tokens": 1000,
            "system": self.personality() + OUTPUT_NOTE,
            "messages": [{"role": "user", "content": prompt}],
            # Une phrase courte : inutile de réfléchir longtemps.
            "extra_body": {"output_config": {"effort": "low"}},
        }
        if (off := thinking_off(self.settings.model)) is not None:
            params["thinking"] = off
        try:
            response = await self._call(params, has_image=False)
        except ClaudeError:
            return None
        self._log_usage([response])
        lines = [line.strip() for line in self._final_text([response]).splitlines() if line.strip()]
        # Un emoji du serveur s'afficherait sous forme de code (<:nom:123…>) : on le retire.
        thought = CUSTOM_EMOJI.sub("", lines[0]).strip("\"'«» ") if lines else ""
        thought = re.sub(r"\s{2,}", " ", thought)
        if not thought or "<" in thought:  # vide, ou balise interne écrite par erreur
            logger.warning("Statut personnalisé inutilisable : %r", thought)
            return None
        return thought[:max_length]

    async def _call(self, params: dict, has_image: bool):
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
            # Toutes les erreurs 400 ont le même type : seul le texte de l'API en donne la cause.
            message = self._api_message(exc).lower()
            if has_image and "image" in message:
                logger.error("Image refusée par l'API : %s", exc)
                raise ClaudeError("Je n'ai pas réussi à lire cette image, essaie avec une autre.")
            if "prompt is too long" in message:
                logger.error("Requête trop longue pour le modèle : %s", exc)
                raise ClaudeError("Ce message est trop long pour moi avec la conversation en cours. "
                                  "Un modérateur peut vider ma mémoire avec /reset.")
            if "credit balance" in message:
                logger.critical("Crédit Anthropic épuisé : %s", exc)
                raise ClaudeError(CONFIG_ERROR)
            logger.critical("Requête refusée par l'API, sans doute à cause d'un réglage du .env "
                            "(MODEL, MAX_TOKENS, THINKING, EFFORT, WEB_SEARCH…) : %s", exc)
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.AuthenticationError:
            logger.critical("Clé API Anthropic invalide : vérifie ANTHROPIC_API_KEY dans le .env.")
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.PermissionDeniedError as exc:
            logger.critical("Accès refusé par l'API (clé sans accès à ce modèle ou à cet outil ?) : %s", exc)
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.NotFoundError as exc:
            logger.critical("Modèle introuvable : vérifie MODEL dans le .env (%s).", exc)
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.APIStatusError as exc:
            logger.error("Erreur de l'API (%s) : %s", exc.status_code, exc)
            raise ClaudeError("Une erreur est survenue de mon côté, réessaie dans un instant.")

    @staticmethod
    def _api_message(exc: anthropic.APIStatusError) -> str:
        """Texte de l'erreur renvoyée par l'API ({"error": {"message": …}})."""
        body = exc.body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            return str(body["error"].get("message", ""))
        return str(exc)

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
