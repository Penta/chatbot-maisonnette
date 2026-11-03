"""Communication avec l'API de Claude (Anthropic)."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .config import Settings
from .emojis import DISCORD_EMOJI, SHORT_EMOJI
from .errors import UserFacingError
from .media import ImageData
from .memory import MAX_NICKNAME_LENGTH, MAX_NICKNAMES, MEMORY_TOOL, MemoryStore, ToolResult
from .models import thinking_off
from .prompts import (CUSTOM_STATUS_PROMPT, CUSTOM_STATUS_REQUESTS, DEFAULT_PERSONALITY, EMOJI_NOTE, GROUP_NOTE,
                      MEMBER_MEMORY_NOTE, MEMORY_NOTE, OUTPUT_NOTE, PRESENCE_NOTE, TOOL_NOTE, WEB_NOTE)
from .text_utils import strip_reasoning

logger = logging.getLogger("chatbot.claude")

# Erreur due à la configuration (clé, crédit, modèle, réglages) : seul l'administrateur peut la corriger.
CONFIG_ERROR = "Je suis mal configuré, préviens l'administrateur du bot."
TRUNCATED_ERROR = ("Ma réponse a été coupée avant que j'aie pu l'écrire. Si cela se répète, "
                   "préviens l'administrateur du bot.")

# Nombre maximal de relances quand l'API met la réponse en pause pendant les recherches.
MAX_CONTINUATIONS = 3


class ClaudeError(UserFacingError):
    """Erreur lors de l'appel à Claude, avec un message affichable sur Discord."""


@dataclass
class Answer:
    text: str
    web_used: bool  # Claude a fait au moins une recherche web ou lu une page


class ClaudeClient:
    def __init__(self, settings: Settings, memory: MemoryStore | None = None):
        self.settings = settings
        self.memory = memory  # None : mémoire des membres désactivée
        # Le SDK réessaie tout seul les erreurs temporaires (surcharge, limite de requêtes…),
        # dans la limite de API_TIMEOUT au total (voir _call).
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.api_timeout,
            max_retries=3,
        )
        self._personality_cache: tuple[float, str] | None = None

    async def close(self) -> None:
        await self._client.close()

    async def check_access(self) -> str | None:
        """Vérifie au démarrage la clé API et CLAUDE_MODEL, par un appel gratuit (aucun token consommé).
        Renvoie le problème à corriger si le bot ne peut pas fonctionner, None sinon. Une panne passagère
        de l'API n'empêche pas le démarrage. Un crédit épuisé ne se voit qu'au premier message."""
        model = self.settings.model
        try:
            await asyncio.wait_for(self._client.models.retrieve(model), min(self.settings.api_timeout, 15))
        except anthropic.AuthenticationError:
            return "clé API Anthropic invalide : vérifie ANTHROPIC_API_KEY dans le .env."
        except anthropic.PermissionDeniedError as exc:
            return f"accès refusé par l'API Anthropic (clé désactivée ou sans accès ?) : {self._api_message(exc)}"
        except anthropic.NotFoundError:
            return f"modèle {model} introuvable chez Anthropic : vérifie CLAUDE_MODEL dans le .env."
        except (anthropic.APIError, asyncio.TimeoutError) as exc:
            logger.warning("Clé API et modèle non vérifiés au démarrage (API injoignable ?) : %s",
                           str(exc) or "délai dépassé")
        return None

    # ------------------------------------------------------------ personnalité

    def personality(self) -> str:
        """Lit personality_prompt.txt (relu automatiquement s'il a été modifié)."""
        path: Path = self.settings.personality_file
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            if self._personality_cache != (0.0, DEFAULT_PERSONALITY):
                logger.warning("%s introuvable : utilisation de la personnalité par défaut.", path)
                self._personality_cache = (0.0, DEFAULT_PERSONALITY)
            return DEFAULT_PERSONALITY
        except OSError as e:
            return self._personality_unreadable(-1.0, f"{path} illisible ({e})")
        if self._personality_cache is not None and self._personality_cache[0] == mtime:
            return self._personality_cache[1]
        if path.is_dir():  # ConfigMap montée comme dossier (Windows lèverait une PermissionError peu parlante)
            return self._personality_unreadable(
                mtime, f"{path} est un dossier : PERSONALITY_FILE doit désigner le fichier lui-même")
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            return self._personality_unreadable(
                mtime, f"{path} n'est pas en UTF-8 : enregistre-le en UTF-8 (Bloc-notes : Enregistrer sous, Encodage)")
        except OSError as e:
            return self._personality_unreadable(mtime, f"{path} illisible ({e})")
        self._personality_cache = (mtime, text or DEFAULT_PERSONALITY)
        logger.info("Personnalité chargée depuis %s", path)
        return self._personality_cache[1]

    def _personality_unreadable(self, mtime: float, reason: str) -> str:
        """Garde la dernière personnalité valide (ou celle par défaut) ; une seule erreur par version du fichier."""
        text = self._personality_cache[1] if self._personality_cache else DEFAULT_PERSONALITY
        if self._personality_cache is None or self._personality_cache[0] != mtime:
            kept = "celle par défaut" if text == DEFAULT_PERSONALITY else "la dernière valide"
            logger.error("%s. Personnalité conservée : %s.", reason, kept)
            self._personality_cache = (mtime, text)
        return text

    # ------------------------------------------------------------------ appel

    @staticmethod
    def build_user_content(parts: list[tuple[str, list[ImageData]]]) -> str | list[dict]:
        """Un ou plusieurs messages à la suite, chacun précédé de ses images (ou de son PDF) éventuels."""
        if not any(images for _, images in parts):
            return "\n".join(text for text, _ in parts)
        content: list[dict] = []
        for text, images in parts:
            for number, image in enumerate(images, 1):
                if len(images) > 1:  # numérotées : Claude peut parler de « la deuxième image »
                    content.append({"type": "text", "text": f"Image {number} :"})
                content.append({
                    "type": "document" if image.is_pdf else "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": base64.standard_b64encode(image.data).decode("ascii"),
                    },
                })
            content.append({"type": "text", "text": text})
        return content

    async def ask(self, history: list[dict], parts: list[tuple[str, list[ImageData]]],
                  emojis: list[str] | None = None, cache_history: bool = True, member_notes: str = "") -> Answer:
        """Envoie l'historique + le ou les nouveaux messages (texte, images ou PDF), et renvoie la réponse de Claude.

        emojis : noms des emojis du serveur que Claude peut écrire (voir EMOJI_NOTE).
        cache_history=False évite d'écrire les messages en cache quand l'appel suivant ne les
        reprendra pas (mentions : contexte différent à chaque fois).
        member_notes : fiches des membres concernés (voir MemoryStore.notes_for), jointes au message
        mais absentes de l'historique."""
        content = self.build_user_content(parts)
        # Les fiches, absentes de l'historique enregistré, viennent après le marqueur de cache :
        # placées avant, elles empêcheraient le message suivant de relire l'historique en cache.
        history_marker = cache_history
        if member_notes:
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if cache_history:
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                history_marker = False  # marqueur déjà placé, avant les fiches
            content.append({"type": "text", "text": member_notes})
        messages = list(history) + [{"role": "user", "content": content}]
        attachments = {("pdf" if image.is_pdf else "image") for _, images in parts for image in images}

        system = self.personality() + MEMORY_NOTE + GROUP_NOTE + OUTPUT_NOTE
        if self.settings.web_search:
            system += WEB_NOTE
        if self.memory:
            system += MEMBER_MEMORY_NOTE.format(member_limit=self.memory.member_limit,
                                                server_limit=self.memory.server_limit,
                                                max_nicknames=MAX_NICKNAMES,
                                                max_nickname_length=MAX_NICKNAME_LENGTH)
        if self.settings.web_search or self.memory:
            system += TOOL_NOTE
        if self.settings.presence:
            system += PRESENCE_NOTE
        if emojis:
            system += EMOJI_NOTE.format(example=emojis[0], names=" ".join(f":{name}:" for name in emojis))
        if self.memory:
            # En dernier : la liste des membres et le fichier général sont ce qui change le plus souvent.
            # Le prompt système n'a pourtant qu'un marqueur de cache : tout changement ici fait réécrire
            # en cache tout le prompt, historique compris, à la requête suivante.
            system += self.memory.system_context()

        # Cache de prompt : les outils et la personnalité, identiques d'un message à l'autre, sont
        # relus depuis le cache à 10 % du prix (pendant 5 minutes, prolongées à chaque message).
        # Le marqueur sur "system" couvre ce bloc fixe ; celui de premier niveau (ou celui placé avant
        # les fiches) couvre en plus l'historique, qui change à chaque message mais reprend le début
        # de la requête précédente.
        # Une écriture en cache coûte 25 % de plus : inutile si elle ne sera jamais relue.
        params: dict = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if history_marker:
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
        tools: list[dict] = []
        if self.settings.web_search:
            # Outils exécutés par Anthropic : leurs appels se font sans nous.
            tools += [
                {"type": "web_search_20260209", "name": "web_search", "max_uses": self.settings.web_max_uses},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": self.settings.web_max_uses},
            ]
        if self.memory:
            tools.append(MEMORY_TOOL)  # exécuté par le bot : voir _run_tools
        if tools:
            params["tools"] = tools

        responses = [await self._call(params, attachments)]
        pauses = operations = 0
        while True:
            last = responses[-1]
            # Lors de longues recherches, l'API peut rendre la main avant la fin (pause_turn) :
            # on renvoie alors sa réponse partielle telle quelle pour qu'elle continue.
            if last.stop_reason == "pause_turn" and pauses < MAX_CONTINUATIONS:
                pauses += 1
                follow_up = []
            # Appel à l'outil « memoire » : le bot l'exécute et renvoie le résultat. Une fois la limite
            # d'opérations atteinte, un dernier tour lui demande de répondre, puis la boucle s'arrête.
            elif last.stop_reason == "tool_use" and self.memory and operations <= self.settings.memory_max_operations:
                results = await self._run_tools(last, operations)
                follow_up = [{"role": "user", "content": results}]
                operations += len(results)
            else:
                break
            # Réponse renvoyée telle quelle, réflexion comprise : l'API l'exige pour continuer.
            messages = messages + [{"role": "assistant", "content": last.content}, *follow_up]
            responses.append(await self._call({**params, "messages": messages}, attachments))

        text = self._final_text(responses)
        web_used = self._log_usage(responses)
        # Sans ce nettoyage, la réflexion affichée resterait en mémoire et le modèle l'imiterait ensuite.
        if (cleaned := strip_reasoning(text)) != text:
            logger.warning("Réflexion écrite dans la réponse, retirée avant l'envoi : %r", text[:500])
            text = cleaned

        stop_reason = responses[-1].stop_reason
        if stop_reason == "max_tokens":
            # Un appel d'outil coupé n'est pas exécuté : sans ce message, la note perdue passerait inaperçue.
            cut = [block.name for block in responses[-1].content if block.type in ("tool_use", "server_tool_use")]
            if cut:
                logger.warning("Appel à l'outil « %s » coupé par MAX_TOKENS (%s), donc pas exécuté : augmente "
                               "MAX_TOKENS dans le .env.", ", ".join(cut), self.settings.max_tokens)
            else:
                logger.warning("Réponse coupée par MAX_TOKENS (%s) : augmente MAX_TOKENS dans le .env "
                               "si cela arrive souvent.", self.settings.max_tokens)
            # Coupée avant son premier mot (réflexion ou appel d'outil trop long) : « […] » seul
            # n'aurait aucun sens sur Discord ni dans l'historique.
            if not text:
                raise ClaudeError(TRUNCATED_ERROR)
            text += " […]"
        elif stop_reason == "pause_turn":
            logger.warning("Recherche web toujours en cours après %s relances : réponse partielle.", MAX_CONTINUATIONS)
        if not text:
            logger.warning("Réponse vide de Claude (stop_reason=%s).", stop_reason)
            raise ClaudeError("Je n'ai rien trouvé à répondre à ça.")
        return Answer(text, web_used)

    async def _run_tools(self, response, done: int) -> list[dict]:
        """Exécute les appels à l'outil « memoire » d'une réponse. Renvoie leurs résultats, à envoyer
        tous dans un même message comme l'exige l'API.
        done : opérations déjà faites pour cette réponse."""
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if done >= self.settings.memory_max_operations:
                result = ToolResult("Limite d'opérations atteinte pour ce message : réponds maintenant, "
                                    "sans rien noter de plus.", is_error=True)
            elif block.name != MEMORY_TOOL["name"] or not isinstance(block.input, dict):
                result = ToolResult(f"Outil inconnu : {block.name}.", is_error=True)
            else:
                result = await self.memory.run(block.input)
            done += 1
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result.content,
                            "is_error": result.is_error})
        return results

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
            response = await self._call(params)
        except ClaudeError:
            return None
        self._log_usage([response])
        lines = [line.strip() for line in self._final_text([response]).splitlines() if line.strip()]
        # Un emoji du serveur s'afficherait sous forme de code (<:nom:123…> ou :nom:) : on le retire.
        thought = SHORT_EMOJI.sub("", DISCORD_EMOJI.sub("", lines[0])).strip("\"'«» ") if lines else ""
        thought = re.sub(r"\s{2,}", " ", thought)
        if not thought or "<" in thought:  # vide, ou balise interne écrite par erreur
            logger.warning("Statut personnalisé inutilisable : %r", thought)
            return None
        return thought[:max_length]

    async def _call(self, params: dict, attachments: set[str] | frozenset[str] = frozenset()):
        """Appel à l'API ; attachments : pièces jointes envoyées ("image", "pdf"), pour les messages d'erreur."""
        try:
            # API_TIMEOUT borne l'attente totale, relances du SDK comprises : une erreur rapide (surcharge)
            # est encore réessayée, mais une requête trop lente n'est pas relancée (recherches refacturées).
            return await asyncio.wait_for(self._client.messages.create(**params), self.settings.api_timeout)
        except anthropic.RateLimitError:
            logger.warning("Limite de requêtes Anthropic atteinte.")
            raise ClaudeError("Je reçois trop de messages en ce moment, réessaie dans une minute.")
        except (anthropic.APITimeoutError, asyncio.TimeoutError):
            logger.warning("Délai dépassé lors de l'appel à Claude (API_TIMEOUT=%g s).", self.settings.api_timeout)
            raise ClaudeError("Je mets trop de temps à répondre, réessaie dans un instant.")
        except anthropic.APIConnectionError as exc:
            logger.error("Connexion à l'API impossible : %s", exc)
            raise ClaudeError("Je n'arrive pas à joindre mon cerveau pour l'instant, réessaie plus tard.")
        except anthropic.BadRequestError as exc:
            # Toutes les erreurs 400 ont le même type : seul le texte de l'API en donne la cause.
            message = self._api_message(exc).lower()
            if "pdf" in attachments and ("pdf" in message or "document" in message):
                logger.error("PDF refusé par l'API : %s", exc)
                raise ClaudeError("Je n'ai pas réussi à lire ce PDF, essaie avec un autre.")
            if "image" in attachments and "image" in message:
                logger.error("Image refusée par l'API : %s", exc)
                raise ClaudeError("Je n'ai pas réussi à lire cette image, essaie avec une autre.")
            if "prompt is too long" in message:
                logger.error("Requête trop longue pour le modèle : %s", exc)
                raise ClaudeError("Ce message est trop long pour moi avec la conversation en cours. "
                                  "Un administrateur peut vider ma mémoire avec /reset.")
            if "credit balance" in message:
                logger.critical("Crédit Anthropic épuisé : %s", exc)
                raise ClaudeError(CONFIG_ERROR)
            logger.critical("Requête refusée par l'API, sans doute à cause d'un réglage du .env "
                            "(CLAUDE_MODEL, MAX_TOKENS, THINKING, EFFORT, WEB_SEARCH…) : %s", exc)
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.AuthenticationError:
            logger.critical("Clé API Anthropic invalide : vérifie ANTHROPIC_API_KEY dans le .env.")
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.PermissionDeniedError as exc:
            logger.critical("Accès refusé par l'API (clé sans accès à ce modèle ou à cet outil ?) : %s", exc)
            raise ClaudeError(CONFIG_ERROR)
        except anthropic.NotFoundError as exc:
            logger.critical("Modèle introuvable : vérifie CLAUDE_MODEL dans le .env (%s).", exc)
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
