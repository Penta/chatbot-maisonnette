import os
import openai
import discord
import aiohttp
import asyncio
import base64
import logging

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Vérifier que les tokens sont récupérés
if DISCORD_TOKEN is None or OPENAI_API_KEY is None:
    raise ValueError("Les tokens ne sont pas définis dans les variables d'environnement.")

# Initialiser les intents
intents = discord.Intents.default()
intents.message_content = True  # Activer l'intent pour les contenus de message

# Initialiser le client Discord avec les intents modifiés
client_discord = discord.Client(intents=intents)

# Initialiser l'API OpenAI avec un client
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# Dictionnaire pour stocker l'historique des conversations pour chaque utilisateur
conversation_history = {}

# L'ID du salon spécifique où le bot est autorisé à répondre
chatgpt_channel_id = 1284699709188997150  # Remplace par l'ID réel de ton salon

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

async def call_openai_api(user_id, user_text, image_data=None):
    # Récupérer l'historique de la conversation pour l'utilisateur
    user_history = conversation_history.get(user_id, [])

    # Préparer le contenu de l'utilisateur
    user_content = [{"type": "text", "text": user_text}]
    if image_data:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })

    # Ajouter le contenu à l'historique
    user_history.append({
        "role": "user",
        "content": user_content
    })

    payload = {
        "model": "gpt-4o",
        "messages": user_history,
        "max_tokens": 500,
        "stop": ["\n"]  # Arrête la réponse à la fin d'une phrase
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
                logging.info(f"Input Tokens: {input_tokens}")
                logging.info(f"Output Tokens: {output_tokens}")
                logging.info(f"Total Tokens: {input_tokens + output_tokens}")
                logging.info(f"Estimated Cost: ${total_cost:.4f}")

                return result
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

@client_discord.event
async def on_ready():
    print(f'Bot connecté en tant que {client_discord.user}')

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
    result = await call_openai_api(message.author.id, user_text, image_data)
    if result:
        reply = result['choices'][0]['message']['content']
        await message.channel.send(reply)

        # Ajouter la réponse du modèle à l'historique
        add_to_conversation_history(message.author.id, {
            "role": "assistant",
            "content": reply
        })

MAX_HISTORY_LENGTH = 50 # Nombre maximum de messages à conserver

def add_to_conversation_history(user_id, new_message):
    if user_id not in conversation_history:
        # Ajouter la personnalité de l'IA en tant que premier message
        conversation_history[user_id] = [
            {
                "role": "system",
                "content": "Tu es une utilisatrice Discord qui parle français."
                           "Tu ne dis jamais que tu es un IA."
                           "Tu ne peux pas poster de message de plusieurs lignes."
                           "Tu peux analyser des images et des pièces jointes aux formats txt, py, html, css et js."
                           "Tu es cynique et sarcastique."
                           "Tu apprécies les compliments."
                           "Réponds sans poser de questions finales comme 'Avez-vous besoin de quelque chose d'autre ?'"
                           "Privilégie des réponses courtes et percutantes."
                           "Tu évites de générer des réponses de plus de 500 tokens."
            }
        ]

    conversation_history[user_id].append(new_message)

    # Limiter la taille de l'historique
    if len(conversation_history[user_id]) > MAX_HISTORY_LENGTH:
        # Garder le premier message de personnalité et les messages les plus récents
        conversation_history[user_id] = conversation_history[user_id][:1] + conversation_history[user_id][-MAX_HISTORY_LENGTH:]

# Démarrer le bot Discord
client_discord.run(DISCORD_TOKEN)
