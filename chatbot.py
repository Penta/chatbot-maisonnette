import os
import base64
import logging
import re
from io import BytesIO
import discord
from dotenv import load_dotenv
from PIL import Image
import emoji
import tiktoken
from openai import AsyncOpenAI, OpenAIError
import json

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
PERSONALITY_PROMPT_FILE = os.getenv('PERSONALITY_PROMPT_FILE', 'personality_prompt.txt')
CONVERSATION_HISTORY_FILE = os.getenv('CONVERSATION_HISTORY_FILE', 'conversation_history.json')

# Initialiser le client OpenAI asynchrone ici
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

BOT_VERSION = "2.2.0"

# Vérifier que les tokens et le prompt de personnalité sont récupérés
if DISCORD_TOKEN is None or OPENAI_API_KEY is None or DISCORD_CHANNEL_ID is None:
    raise ValueError("Les tokens ou l'ID du canal ne sont pas définis dans les variables d'environnement.")

if not os.path.isfile(PERSONALITY_PROMPT_FILE):
    raise FileNotFoundError(f"Le fichier de prompt de personnalité '{PERSONALITY_PROMPT_FILE}' est introuvable.")

# Lire le prompt de personnalité depuis le fichier
with open(PERSONALITY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    PERSONALITY_PROMPT = f.read().strip()

# Log configuration
log_format = '%(asctime)-13s : %(name)-15s : %(levelname)-8s : %(message)s'
logging.basicConfig(handlers=[logging.FileHandler("./chatbot.log", 'a', 'utf-8')], format=log_format, level="INFO")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger("chatbot")
logger.setLevel("INFO")

logging.getLogger('').addHandler(console)

httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)

# Initialiser les intents
intents = discord.Intents.default()
intents.message_content = True  # Activer l'intent pour les contenus de message

# Liste pour stocker l'historique des conversations
conversation_history = []

def load_conversation_history():
    global conversation_history
    if os.path.isfile(CONVERSATION_HISTORY_FILE):
        try:
            with open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                conversation_history = json.load(f)
            logger.info(f"Historique chargé depuis {CONVERSATION_HISTORY_FILE}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique : {e}")
            conversation_history = []
    else:
        logger.info(f"Aucun fichier d'historique trouvé. Un nouveau fichier sera créé à {CONVERSATION_HISTORY_FILE}")

def has_text(text):
    """
    Détermine si le texte fourni est non vide après suppression des espaces.
    """
    return bool(text.strip())

def save_conversation_history():
    try:
        with open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'historique : {e}")

# Charger l'encodeur pour le modèle GPT-4o mini
encoding = tiktoken.get_encoding("o200k_base")

# Convertir l'ID du channel en entier
try:
    chatgpt_channel_id = int(DISCORD_CHANNEL_ID)
except ValueError:
    raise ValueError("L'ID du channel Discord est invalide. Assurez-vous qu'il s'agit d'un entier.")

class MyDiscordClient(discord.Client):
    async def close(self):
        global openai_client
        if openai_client is not None:
            await openai_client.close()
            openai_client = None
        await super().close()

# Initialiser le client Discord avec les intents modifiés
client_discord = MyDiscordClient(intents=intents)

# Appeler la fonction pour charger l'historique au démarrage
load_conversation_history()

def resize_image(image_bytes, mode='high', attachment_filename=None):
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            original_format = img.format  # Store the original format

            if mode == 'high':
                # Redimensionner pour le mode haute fidélité
                img.thumbnail((2000, 2000))
                if min(img.size) < 768:
                    scale = 768 / min(img.size)
                    new_size = tuple(int(x * scale) for x in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
            elif mode == 'low':
                # Redimensionner pour le mode basse fidélité
                img = img.resize((512, 512))

            buffer = BytesIO()

            img_format = img.format
            if not img_format:
                if attachment_filename:
                    _, ext = os.path.splitext(attachment_filename)
                    ext = ext.lower()
                    format_mapping = {
                        '.jpg': 'JPEG',
                        '.jpeg': 'JPEG',
                        '.png': 'PNG',
                        '.gif': 'GIF',
                        '.bmp': 'BMP',
                        '.tiff': 'TIFF'
                    }
                    img_format = format_mapping.get(ext, 'PNG')
                else:
                    img_format = 'PNG'

            img.save(buffer, format=img_format)
            return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise

def is_long_special_text(text):
    # Vérifier que le texte est bien une chaîne de caractères
    if not isinstance(text, str):
        logger.error(f"Erreur : Le contenu n'est pas une chaîne valide. Contenu : {text}")
        return False

    # Compter le nombre de tokens dans le texte
    token_count = len(encoding.encode(text))

    # Définir un seuil pour considérer le texte comme long
    if token_count > 200:
        logger.info("Texte long détecté : %d tokens", token_count)
        return True
    return False

def extract_text_from_message(message):
    content = message.get("content", "")
    if isinstance(content, list):
        # Extraire le texte de chaque élément de la liste
        texts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    texts.append(text)
        return ' '.join(texts)
    elif isinstance(content, str):
        return content
    else:
        return ""

def calculate_cost(usage, model='gpt-4o-mini'):
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)

    # Définir les tarifs par modèle
    model_costs = {
        'gpt-4o': {
            'input_rate': 5.00 / 1_000_000,    # 5$ pour 1M tokens d'entrée
            'output_rate': 15.00 / 1_000_000  # 15$ pour 1M tokens de sortie
        },
        'gpt-4o-mini': {
            'input_rate': 0.150 / 1_000_000,   # 0.150$ pour 1M tokens d'entrée
            'output_rate': 0.600 / 1_000_000   # 0.600$ pour 1M tokens de sortie
        }
    }

    # Obtenir les tarifs du modèle spécifié
    if model not in model_costs:
        logger.warning(f"Modèle inconnu '{model}'. Utilisation des tarifs par défaut pour 'gpt-4o-mini'.")
        model = 'gpt-4o-mini'

    input_rate = model_costs[model]['input_rate']
    output_rate = model_costs[model]['output_rate']

    # Calculer les coûts
    input_cost = input_tokens * input_rate
    output_cost = output_tokens * output_rate
    total_cost = input_cost + output_cost

    return input_tokens, output_tokens, total_cost

def is_relevant_message(message):
    content = message["content"]

    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content if 'text' in part)

    if len(content.strip()) < 5:
        return False

    discord_emoji_pattern = r'<a?:\w+:\d+>'

    def is_discord_emoji(part):
        return bool(re.fullmatch(discord_emoji_pattern, part))

    tokens = re.split(discord_emoji_pattern, content)
    emojis_only = True
    standard_emojis = [char for char in content if emoji.is_emoji(char)]
    discord_emojis = re.findall(discord_emoji_pattern, content)

    text_without_emojis = re.sub(discord_emoji_pattern, '', content)
    for char in text_without_emojis:
        if not char.isspace() and not emoji.is_emoji(char):
            emojis_only = False
            break

    if len(standard_emojis) + len(discord_emojis) == 0:
        emojis_only = False

    if emojis_only and len(content.strip()) > 0:
        return False

    return True

async def call_gpt4o_for_image_analysis(image_data, user_text=None, detail='high'):
    try:
        # Préparer la requête pour GPT-4o
        if user_text:
            prompt = f"Analyse cette image en tenant compte de la description suivante : \"{user_text}\"."
        else:
            prompt = "Analyse cette image s'il te plaît."

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
            logging.info(f"Analyse de l'image par GPT-4o : {analysis}")

            # Calcul et affichage du coût
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
                input_tokens, output_tokens, total_cost = calculate_cost(usage, model='gpt-4o')
                logging.info(f"Coût de l'analyse de l'image : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens}")
            else:
                logging.warning("Informations d'utilisation non disponibles pour le calcul du coût.")

            return analysis
        else:
            return None 

    except OpenAIError as e:
        logger.error(f"Erreur lors de l'analyse de l'image avec GPT-4o: {e}")
        return None

async def call_gpt4o_mini_with_analysis(analysis_text, user_name, user_question, has_text):
    try:
        # Préparer le message avec le prompt de personnalité et l'historique
        prompt_personality = {"role": "system", "content": PERSONALITY_PROMPT}

        # Préparer le contexte de l'analyse
        analysis_message = {
            "role": "system",
            "content": (
                f"L'analyse de l'image fournie est la suivante :\n{analysis_text}\n\n"
            )
        }

        if has_text:
            # Préparer le message utilisateur avec le texte
            user_message = {
                "role": "user",
                "content": (
                    f"{user_name} a écrit : '{user_question}'.\n"
                    "Veuillez répondre en vous basant uniquement sur l'analyse fournie ci-dessus."
                )
            }
        else:
            # Préparer une instruction pour commenter l'image
            user_message = {
                "role": "user",
                "content": (
                    f"{user_name} a partagé une image sans texte additionnel.\n"
                    "Veuillez commenter cette image en vous basant uniquement sur l'analyse fournie ci-dessus."
                )
            }

        # Assembler les messages
        messages = [prompt_personality, analysis_message] + conversation_history + [user_message]

        # Appel à GPT-4o Mini pour réagir à la question et à l'analyse
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=450
        )

        if response:
            reply = response.choices[0].message.content
            logging.info(f"Réponse de GPT-4o Mini : {reply}")
            return reply
        else:
            return None

    except OpenAIError as e:
        logger.error(f"Erreur lors de la génération de réponse avec GPT-4o Mini: {e}")
        return None

async def read_text_file(attachment):
    file_bytes = await attachment.read()
    return file_bytes.decode('utf-8')

async def encode_image_from_attachment(attachment, mode='high'):
    image_data = await attachment.read()
    resized_image = resize_image(image_data, mode=mode, attachment_filename=attachment.filename)
    return base64.b64encode(resized_image).decode('utf-8')

async def summarize_text(text, max_tokens=50):
    summary_prompt = f"Résumé :\n\n{text}\n\nRésumé:"
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=max_tokens  # Limitez les tokens pour obtenir un résumé court
        )
        summary = response.choices[0].message.content.strip()
        if hasattr(response, 'usage'):
            usage_dict = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
        else:
            usage_dict = {}
        return summary, usage_dict
    except OpenAIError as e:
        logger.error(f"Error summarizing text: {e}")
        return text, {}
    except AttributeError as e:
        logger.error(f"Attribute error during summarization: {e}")
        return text, {}

async def call_openai_api(user_text, user_name, image_data=None, detail='high'):

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

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history + [message_to_send],
            max_tokens=400,
            temperature=1.0
        )

        if response:
            reply = response.choices[0].message.content

        # Ajouter le message de l'utilisateur à l'historique global, mais uniquement s'il ne s'agit pas d'une image
        if image_data is None:
            await add_to_conversation_history(message_to_send)

        # Ajouter la réponse de l'IA directement à l'historique
        await add_to_conversation_history({
            "role": "assistant",
            "content": reply
        })

        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            input_tokens, output_tokens, total_cost = calculate_cost({
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens
            })

        # Afficher dans la console
        logging.info(f"Coût de la réponse : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens} / Total: {input_tokens + output_tokens}")

        return response
    except OpenAIError as e:
        logger.error(f"Error calling OpenAI API: {e}")
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
    return None

@client_discord.event
async def on_ready():
    logger.info(f'Bot connecté en tant que {client_discord.user}')

    # Ajouter la personnalité de l'IA à l'historique au démarrage
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": PERSONALITY_PROMPT
        })
        save_conversation_history()

    # Envoyer un message de version dans le canal Discord
    channel = client_discord.get_channel(chatgpt_channel_id)
    if channel:
        try:
            embed = discord.Embed(
                title="Bot Démarré",
                description=f"🎉 Le ChatBot est en ligne ! Version {BOT_VERSION}",
                color=0x00ff00  # Vert
            )
            await channel.send(embed=embed)
            logger.info(f"Message de connexion envoyé dans le canal ID {chatgpt_channel_id}")
        except discord.Forbidden:
            logger.error(f"Permissions insuffisantes pour envoyer des messages dans le canal ID {chatgpt_channel_id}.")
        except discord.HTTPException as e:
            logger.error(f"Erreur lors de l'envoi du message de connexion : {e}")
    else:
        logger.error(f"Canal avec ID {chatgpt_channel_id} non trouvé.")

@client_discord.event
async def on_message(message):
    global conversation_history

    # Vérifier si le message provient du canal autorisé
    if message.channel.id != chatgpt_channel_id:
        return

    # Ignorer les messages du bot lui-même
    if message.author == client_discord.user:
        return

    user_text = message.content.strip()
    image_data = None
    file_content = None
    attachment_filename = None

    # Vérifier si le message est la commande de réinitialisation
    if user_text.lower() == "!reset_history":
        # Vérifier si l'utilisateur a les permissions administratives
        if not message.author.guild_permissions.administrator:
            await message.channel.send("❌ Vous n'avez pas la permission d'utiliser cette commande.")
            return

        # Réinitialiser l'historique en conservant uniquement le prompt de personnalité
        conversation_history = [{
            "role": "system",
            "content": PERSONALITY_PROMPT
        }]
        save_conversation_history()
        await message.channel.send("✅ L'historique des conversations a été réinitialisé.")
        logger.info(f"Historique des conversations réinitialisé par {message.author}.")
        return  # Arrêter le traitement du message après la réinitialisation

    # Extensions de fichiers autorisées
    allowed_extensions = ['.txt', '.py', '.html', '.css', '.js']

    # Variables pour stocker si le message contient une image et/ou un fichier
    has_image = False
    has_file = False

    # Vérifier s'il y a une pièce jointe
    if message.attachments:
        for attachment in message.attachments:
            # Vérifier si c'est un fichier avec une extension autorisée
            if any(attachment.filename.endswith(ext) for ext in allowed_extensions):
                file_content = await read_text_file(attachment)
                attachment_filename = attachment.filename
                break
            # Vérifier si c'est une image
            elif attachment.content_type in ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff']:
                image_data = await encode_image_from_attachment(attachment, mode='high')
                break

    # Si une image est présente, la traiter
    if image_data:
        has_user_text = has_text(user_text)
        user_text_to_use = user_text if has_user_text else None

        # Étape 1 : GPT-4o analyse l'image, potentiellement guidée par le texte de l'utilisateur
        analysis = await call_gpt4o_for_image_analysis(image_data, user_text=user_text_to_use)

        if analysis:
            # Étape 2 : GPT-4o Mini réagit à la question et à l'analyse
            reply = await call_gpt4o_mini_with_analysis(analysis, message.author.name, user_text, has_user_text)
            if reply:
                await message.channel.send(reply)

                # **Ajout des messages à l'historique**
                # Créer un message utilisateur modifié indiquant qu'une image a été postée
                if has_user_text:
                    user_message_content = f"{user_text} (a posté une image.)"
                else:
                    user_message_content = "Une image a été postée."

                user_message = {
                    "role": "user",
                    "content": user_message_content
                }

                # Ajouter le message utilisateur à l'historique
                await add_to_conversation_history(user_message)

                # Créer le message assistant avec la réponse de GPT-4o Mini
                assistant_message = {
                    "role": "assistant",
                    "content": reply
                }

                # Ajouter le message assistant à l'historique
                await add_to_conversation_history(assistant_message)
            else:
                await message.channel.send("Désolé, je n'ai pas pu générer une réponse.")
        else:
            await message.channel.send("Désolé, je n'ai pas pu analyser l'image.")
        # Après traitement de l'image, ne pas continuer
        return

    # Ajouter le contenu du fichier à la requête si présent
    if file_content:
        user_text += f"\nContenu du fichier {attachment.filename}:\n{file_content}"

    # Vérifier si le texte n'est pas vide après ajout du contenu du fichier
    if not has_text(user_text):
        return  # Ne pas appeler l'API si le texte est vide

    # Appeler l'API OpenAI
    result = await call_openai_api(user_text, message.author.name, image_data)
    if result:
        reply = result.choices[0].message.content
        await message.channel.send(reply)

async def add_to_conversation_history(new_message):

    # Exclure les messages d'analyse de l'image
    if new_message.get("role") == "system" and "L'analyse de l'image fournie est la suivante :" in new_message.get("content", ""):
        return  # Ne pas ajouter à l'historique

    # Extraire le texte du message
    if isinstance(new_message["content"], list) and len(new_message["content"]) > 0:
        content_text = new_message["content"][0].get("text", "")
    else:
        content_text = new_message.get("content", "")

    if not isinstance(content_text, str):
        logger.error(f"Erreur : Le contenu n'est pas une chaîne valide. Contenu : {content_text}")
        return

    if is_long_special_text(content_text):
        summary, usage = await summarize_text(content_text)
        new_message = {
            "role": new_message["role"],
            "content": summary
        }

        # Inclure le coût du résumé dans le calcul total
        input_tokens, output_tokens, total_cost = calculate_cost(usage)
        logging.info(f"Coût du résumé : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens} / Total: {input_tokens + output_tokens}")

    # Filtrer les messages pertinents pour l'historique
    if is_relevant_message(new_message):
        # Ajouter le message à l'historique
        conversation_history.append(new_message)
        save_conversation_history()

    # Synthétiser les messages les plus anciens si l'historique est trop long
    if len(conversation_history) > 1000:
        # Synthétiser les 20 plus anciens messages (exclure la personnalité et les 10 plus récents)
        messages_to_summarize = conversation_history[1:21]  # Exclure le premier (personnalité)
        texts = [extract_text_from_message(msg) for msg in messages_to_summarize]
        texts = [text for text in texts if text]

        combined_text = ' '.join(texts)

        combined_token_count = len(encoding.encode(combined_text))
        if combined_token_count > 15000:
            encoded_text = encoding.encode(combined_text)
            truncated_text = encoding.decode(encoded_text[:500])
            combined_text = truncated_text
            logger.info(f"Combined text tronqué à 15 000 tokens.")

        synthesized_summary, usage = await summarize_text(combined_text, max_tokens=400)

        # Calculer le coût de la synthèse
        input_tokens, output_tokens, total_cost = calculate_cost(usage)
        logging.info(f"1000 messages dans l'historique. Synthèse effectuée. Coût : ${total_cost:.4f} / Input: {input_tokens} / Output: {output_tokens} / Total: {input_tokens + output_tokens}")

        # Remplacer l'ancienne synthèse par la nouvelle
        # Conserver la personnalité et la nouvelle synthèse
        conversation_history[:] = [conversation_history[0], {"role": "system", "content": synthesized_summary}] + conversation_history[21:]
        save_conversation_history()

# Démarrer le bot Discord
client_discord.run(DISCORD_TOKEN)
