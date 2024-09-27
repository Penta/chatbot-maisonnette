# Initialiser le client OpenAI et définir les fonctions d'interaction avec l'API OpenAI.

from openai import AsyncOpenAI, OpenAIError
from config import OPENAI_API_KEY, PERSONALITY_PROMPT
from logger import logger
from utils import calculate_cost
from history import add_to_conversation_history

# Initialiser le client OpenAI asynchrone
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

IMAGE_ANALYSIS_MARKER = "__IMAGE_ANALYSIS__:"

async def call_gpt4o_for_image_analysis(image_data, user_text=None, detail='high'):
    try:
        # Préparer le prompt
        if user_text:
            prompt = (
                "Tu es un expert en analyse d'images et de textes. "
                "On te présente une image ou un texte qui pourrait contenir des informations importantes. "
                f"Voici ce que l'on te décrit : \"{user_text}\". "
                "Analyse chaque détail de manière méticuleuse. "
                "Si l'image montre un environnement sans personnage, décris minutieusement les objets, leur disposition, les couleurs, textures, formes, et tout autre élément notable. "
                "Si du texte est présent, analyse chaque mot attentivement : style, mise en page, ou tout détail subtil qui pourrait en révéler plus sur le contexte ou l'intention. "
                "Si des personnages sont présents, décris-les avec précision : leur posture, apparence physique, vêtements, et fais une estimation de leurs mensurations (taille, tour de poitrine, taille, hanches, etc.). "
                "Sois attentif aux expressions et aux petits détails dans leur attitude ou apparence qui pourraient donner des indications supplémentaires. "
                "N'oublie aucun détail, car chaque aspect pourrait être révélateur dans cette analyse."
            )
        else:
            prompt = (
                "Tu es un expert en analyse d'images et de textes. "
                "On te présente une image ou un texte qui pourrait contenir des informations importantes. "
                "Analyse chaque détail de manière méticuleuse. "
                "Si l'image montre un environnement sans personnage, décris minutieusement les objets, leur disposition, les couleurs, textures, formes, et tout autre élément notable. "
                "Si du texte est présent, analyse chaque mot attentivement : style, mise en page, ou tout détail subtil qui pourrait en révéler plus sur le contexte ou l'intention. "
                "Si des personnages sont présents, décris-les avec précision : leur posture, apparence physique, vêtements, et fais une estimation de leurs mensurations (taille, tour de poitrine, taille, hanches, etc.). "
                "Sois attentif aux expressions et aux petits détails dans leur attitude ou apparence qui pourraient donner des indications supplémentaires. "
                "N'oublie aucun détail, car chaque aspect pourrait être révélateur dans cette analyse."
            )

        message_to_send = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},   
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": detail
                    }
                }
            ]
        }

        # Appel à GPT-4o
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[message_to_send],
            max_tokens=4096
        )

        if response:
            analysis = response.choices[0].message.content
            logger.info(f"Analyse de l'image par GPT-4o : {analysis}")

            # Calcul et affichage du coût
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
                input_tokens, output_tokens, total_cost = calculate_cost(usage, model='gpt-4o')
                logger.info(f"Coût de l'analyse de l'image : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens}")
            else:
                logger.warning("Informations d'utilisation non disponibles pour le calcul du coût.")

            # Ajouter l'analyse à l'historique avec le marqueur
            analysis_message = {
            	"role": "system",
            	"content": f"{IMAGE_ANALYSIS_MARKER}{analysis}"
            }
            await add_to_conversation_history(analysis_message)

            return analysis
        else:
            return None 

    except OpenAIError as e:
        logger.error(f"Erreur lors de l'analyse de l'image avec GPT-4o: {e}")
        return None

async def call_gpt4o_mini_with_analysis(analysis_text, user_name, user_question, has_text, conversation_history):
    try:
        # Préparer le message avec le prompt de personnalité et l'analyse
        messages = [
            {"role": "system", "content": PERSONALITY_PROMPT},
            {
                "role": "system",
                "content": f"L'analyse de l'image fournie est la suivante :\n{analysis_text}\n\n"
            }
        ]

        if has_text:
            # Préparer le message utilisateur avec le texte
            user_message = {
                "role": "user",
                "content": (
                    "Tu es Rika, une femme de 24 ans et membre à part entière du serveur Discord sur lequel tu te trouves. "
                    f"{user_name} a posté un message contenant une image et a écrit avec : '{user_question}'.\n"
                    "Réponds à l'utilisateur en te basant sur l'analyse, avec ta personnalité. "
                    "Ne mentionne pas explicitement que l'analyse est pré-existante, fais comme si tu l'avais faite toi-même."
                )
            }
        else:
            # Préparer une instruction pour commenter l'image sans texte
            user_message = {
                "role": "user",
                "content": (
                    "Tu es Rika, une femme de 24 ans et membre à part entière du serveur Discord sur lequel tu te trouves. "
                    f"{user_name} a partagé une image sans texte additionnel.\n"
                    "Commente l'image en te basant sur l'analyse, avec ta personnalité. "
                    "Ne mentionne pas que l'analyse a été fournie à l'avance, réagis comme si tu l'avais toi-même effectuée."
                )
            }

        # Inclure l'historique de conversation
        messages += conversation_history
        messages.append(user_message)

        # Appel à GPT-4o Mini pour répondre
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=450
        )

        if response:
            reply = response.choices[0].message.content

            # Calculer et enregistrer le coût de la réponse de GPT-4o Mini
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
                input_tokens, output_tokens, total_cost = calculate_cost(usage, model='gpt-4o-mini')
                logger.info(f"Coût de la réponse de GPT-4o Mini : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens}")
            else:
                logger.warning("Informations d'utilisation non disponibles pour le calcul du coût de GPT-4o Mini.")

            return reply
        else:
            return None

    except OpenAIError as e:
        logger.error(f"Erreur lors de la génération de réponse avec GPT-4o Mini: {e}")
        return None

async def call_openai_api(user_text, user_name, conversation_history, image_data=None, detail='high'):
    try:
        # Préparer le contenu pour l'appel API
        message_to_send = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{user_name} dit : {user_text}"}
            ]
        }

        # Inclure l'image dans l'appel API courant
        if image_data:
            message_to_send["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": detail
                }
            })

        # Assembler les messages avec le prompt de personnalité en premier
        messages = [
            {"role": "system", "content": PERSONALITY_PROMPT}
        ] + conversation_history + [message_to_send]

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=1.0
        )

        if response:
            reply = response.choices[0].message.content

            # Calculer et enregistrer le coût de la réponse de GPT-4o Mini
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
                input_tokens, output_tokens, total_cost = calculate_cost(usage, model='gpt-4o-mini')
                logger.info(f"Coût de la réponse : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens} / Total: {input_tokens + output_tokens}")
            else:
                logger.warning("Informations d'utilisation non disponibles pour le calcul du coût.")

            return response
        else:
            return None

    except OpenAIError as e:
        logger.error(f"Error calling OpenAI API: {e}")
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
    return None
