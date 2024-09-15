import os
import openai
import discord
import aiohttp
import asyncio
import base64
import logging
import re
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')

# Chemin vers le fichier de prompt de personnalité
PERSONALITY_PROMPT_FILE = os.getenv('PERSONALITY_PROMPT_FILE', 'personality_prompt.txt')

# Vérifier que les tokens et le prompt de personnalité sont récupérés
if DISCORD_TOKEN is None or OPENAI_API_KEY is None or DISCORD_CHANNEL_ID is None:
    raise ValueError("Les tokens ou l'ID du canal ne sont pas définis dans les variables d'environnement.")

if not os.path.isfile(PERSONALITY_PROMPT_FILE):
    raise FileNotFoundError(f"Le fichier de prompt de personnalité '{PERSONALITY_PROMPT_FILE}' est introuvable.")

# Lire le prompt de personnalité depuis le fichier
with open(PERSONALITY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    PERSONALITY_PROMPT = f.read().strip()

# Log configuration
log_format='%(asctime)-13s : %(name)-15s : %(levelname)-8s : %(message)s'
logging.basicConfig(handlers=[logging.FileHandler("./chatbot.log", 'a', 'utf-8')], format=log_format, level="INFO")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger("chatbot")
logger.setLevel("INFO")

logging.getLogger('').addHandler(console)

# Initialiser les intents
intents = discord.Intents.default()
intents.message_content = True  # Activer l'intent pour les contenus de message

# Initialiser le client Discord avec les intents modifiés
client_discord = discord.Client(intents=intents)

# Initialiser l'API OpenAI avec un client
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# Liste pour stocker l'historique des conversations
conversation_history = []

# Convertir l'ID du channel en entier
chatgpt_channel_id = int(DISCORD_CHANNEL_ID)

def is_ascii_art(text):
    # Définir un seuil pour la longueur d'une séquence de caractères spéciaux
    threshold_length = 10
    # Chercher des séquences de caractères spéciaux
    special_char_sequences = re.findall(r'[^a-zA-Z0-9\s]{' + str(threshold_length) + ',}', text)

    # Si on trouve une séquence de caractères spéciaux longue, c'est probablement un dessin ASCII
    if any(len(seq) >= threshold_length for seq in special_char_sequences):
        return True
    return False

def is_long_special_text(text):
    # Définir un seuil pour considérer le texte comme long et contenant beaucoup de caractères spéciaux
    special_char_count = len(re.findall(r'[^\w\s]', text))
    if len(text) > 1200 and special_char_count > 200:
        return True
    return False

def calculate_cost(usage):
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)

    # Coûts estimés
    input_cost = input_tokens / 1_000_000 * 5.00  # 5$ pour 1M tokens d'entrée
    output_cost = output_tokens / 1_000_000 * 15.00  # 15$ pour 1M tokens de sortie
    total_cost = input_cost + output_cost

    return input_tokens, output_tokens, total_cost

async def read_text_file(attachment):
    # Télécharger et lire le contenu du fichier texte
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            return await resp.text()

async def encode_image_from_attachment(attachment):
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            image_data = await resp.read()
            return base64.b64encode(image_data).decode('utf-8')

async def call_openai_api(user_text, user_name, image_data=None):

    # Préparer le contenu pour l'appel API
    message_to_send = {
        "role": "user",
        "content": [{"type": "text", "text": f"{user_name} dit : {user_text}"}]
    }

    # Inclure l'image dans l'appel API courant
    if image_data:
        message_to_send["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })

    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": PERSONALITY_PROMPT
        })

    # Ajouter le message de l'utilisateur à l'historique global, mais uniquement s'il ne s'agit pas d'une image ou d'ASCII art
    if image_data is None and not is_ascii_art(user_text):
        conversation_history.append(message_to_send)

    payload = {
        "model": "gpt-4o",
        "messages": conversation_history,
        "max_tokens": 500
    }

    headers = {
       "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                result = await resp.json()
                if resp.status != 200:
                    raise ValueError(f"API Error: {result.get('error', {}).get('message', 'Unknown error')}")

                # Calculer les coûts
                usage = result.get('usage', {})
                input_tokens, output_tokens, total_cost = calculate_cost(usage)

                # Afficher dans la console
                logging.info(f"Estimated Cost: ${total_cost:.4f} / Input Tokens: {input_tokens} / Output Tokens: {output_tokens} / Total Tokens: {input_tokens + output_tokens}")

                return result
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

@client_discord.event
async def on_message(message):
    # Vérifier si le message provient du canal autorisé
    if message.channel.id != chatgpt_channel_id:
        return

    # Vérifier si l'auteur du message est le bot lui-même
    if message.author == client_discord.user:
        return

    user_text = message.content.strip()
    image_data = None
    file_content = None

    # Extensions de fichiers autorisées
    allowed_extensions = ['.txt', '.py', '.html', '.css', '.js']

    # Vérifier s'il y a une pièce jointe
    if message.attachments:
        for attachment in message.attachments:
            # Vérifier si c'est un fichier avec une extension autorisée
            if any(attachment.filename.endswith(ext) for ext in allowed_extensions):
                file_content = await read_text_file(attachment)
                break
            # Vérifier si c'est une image
            elif attachment.content_type.startswith('image/'):
                image_data = await encode_image_from_attachment(attachment)
                break

    # Ajouter le contenu du fichier à la requête si présent
    if file_content:
        user_text += f"\nContenu du fichier {attachment.filename}:\n{file_content}"

    # Appeler l'API OpenAI
    result = await call_openai_api(user_text, message.author.name, image_data)
    if result:
        reply = result['choices'][0]['message']['content']
        await message.channel.send(reply)

        # Ajouter la réponse du modèle à l'historique
        # Ne pas ajouter à l'historique si c'est un dessin ASCII ou une image
        if image_data is None and not is_ascii_art(user_text):
            add_to_conversation_history({
                "role": "assistant",
                "content": reply
            })

MAX_HISTORY_LENGTH = 50 # Nombre maximum de messages à conserver

# Liste pour stocker les indices des messages longs et spéciaux
temporary_messages = []

def add_to_conversation_history(new_message):
    # Ajouter la personnalité de l'IA en tant que premier message
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": PERSONALITY_PROMPT
        })

    # Ajouter le message à l'historique
    conversation_history.append(new_message)

    # Vérifier si le message est long et contient beaucoup de caractères spéciaux
    if new_message["role"] == "user" and is_long_special_text(new_message["content"][0]["text"]):
        # Ajouter l'index de ce message dans la liste des messages temporaires
        temporary_messages.append(len(conversation_history) - 1)

    # Limiter la taille de l'historique
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        # Garder le premier message de personnalité et les messages les plus récents
        conversation_history[:] = conversation_history[:1] + conversation_history[-MAX_HISTORY_LENGTH:]

    # Supprimer les messages temporaires après dix messages
    if len(temporary_messages) > 0:
        for index in reversed(temporary_messages):
            # Supprimer le message s'il a été dans l'historique pendant dix messages ou plus
            if len(conversation_history) - index > 10:
                del conversation_history[index]
                temporary_messages.remove(index)

# Démarrer le bot Discord
client_discord.run(DISCORD_TOKEN)
