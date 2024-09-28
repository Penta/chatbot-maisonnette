import os
import mysql.connector
from mysql.connector import Error
import base64
import json
import logging
import re
from io import BytesIO

import discord
from discord.ext import commands
from dotenv import load_dotenv
from PIL import Image
import tiktoken
from openai import AsyncOpenAI, OpenAIError

# ================================
# Configuration et Initialisation
# ================================

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
PERSONALITY_PROMPT_FILE = os.getenv('PERSONALITY_PROMPT_FILE', 'personality_prompt.txt')
CONVERSATION_HISTORY_FILE = os.getenv('CONVERSATION_HISTORY_FILE', 'conversation_history.json')
BOT_NAME = os.getenv('BOT_NAME', 'ChatBot')
BOT_VERSION = "2.6.0"

# Validation des variables d'environnement
required_env_vars = {
    'DISCORD_TOKEN': DISCORD_TOKEN,
    'OPENAI_API_KEY': OPENAI_API_KEY,
    'DISCORD_CHANNEL_ID': DISCORD_CHANNEL_ID
}

missing_vars = [var for var, val in required_env_vars.items() if val is None]
if missing_vars:
    raise ValueError(f"Les variables d'environnement suivantes ne sont pas définies: {', '.join(missing_vars)}")

# Vérification de l'existence du fichier de prompt de personnalité
if not os.path.isfile(PERSONALITY_PROMPT_FILE):
    raise FileNotFoundError(f"Le fichier de prompt de personnalité '{PERSONALITY_PROMPT_FILE}' est introuvable.")

# Lire le prompt de personnalité depuis le fichier
with open(PERSONALITY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    PERSONALITY_PROMPT = f.read().strip()

# Initialiser le client OpenAI asynchrone
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Configurer les logs
LOG_FORMAT = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
logging.basicConfig(
    handlers=[
        logging.FileHandler("./chatbot.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    format=LOG_FORMAT,
    level=logging.INFO
)
logger = logging.getLogger(BOT_NAME)
logger.setLevel(logging.INFO)

# Réduire le niveau de log pour certaines librairies
logging.getLogger('httpx').setLevel(logging.WARNING)

# Initialiser les intents Discord
intents = discord.Intents.default()
intents.message_content = True

# Initialiser le client Discord
class MyDiscordClient(discord.Client):
    def __init__(self, **options):
        super().__init__(**options)

    async def close(self):
        if openai_client:
            await openai_client.close()
        await super().close()

client_discord = MyDiscordClient(intents=intents)

# Convertir l'ID du canal Discord en entier
try:
    chatgpt_channel_id = int(DISCORD_CHANNEL_ID)
except ValueError:
    raise ValueError("L'ID du canal Discord est invalide. Assurez-vous qu'il s'agit d'un entier.")

# ========================
# Configuration de la base de données
# ========================

def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        if connection.is_connected():
            logger.info("Connexion réussie à MariaDB")
            return connection
    except Error as e:
        logger.error(f"Erreur de connexion à MariaDB: {e}")
        return None

# ========================
# Gestion du chargement et de la sauvegarde de l'Historique
# ========================

conversation_history = []
last_analysis_index = None
messages_since_last_analysis = 0

def load_conversation_history(db_connection):
    global conversation_history
    try:
        cursor = db_connection.cursor(dictionary=True)
        cursor.execute("SELECT role, content FROM conversation_history ORDER BY id ASC")
        rows = cursor.fetchall()
        conversation_history = [row for row in rows if not (row['role'] == "system" and row['content'] == PERSONALITY_PROMPT)]
        logger.info("Historique chargé depuis la base de données")
    except Error as e:
        logger.error(f"Erreur lors du chargement de l'historique depuis la base de données: {e}")
        conversation_history = []
    finally:
        cursor.close()

def save_message_to_db(db_connection, role, content):
    try:
        cursor = db_connection.cursor()
        sql = "INSERT INTO conversation_history (role, content) VALUES (%s, %s)"
        cursor.execute(sql, (role, json.dumps(content) if isinstance(content, (dict, list)) else content))
        db_connection.commit()
        logger.debug(f"Message sauvegardé dans la base de données: {role} - {content[:50]}...")
    except Error as e:
        logger.error(f"Erreur lors de la sauvegarde du message dans la base de données: {e}")
    finally:
        cursor.close()

# ====================
# Fonctions Utilitaires
# ====================

def has_text(text):
    """Détermine si le texte fourni est non vide après suppression des espaces."""
    return bool(text.strip())

def resize_image(image_bytes, mode='high', attachment_filename=None):
    """Redimensionne l'image selon le mode spécifié."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            original_format = img.format  # Stocker le format original

            if mode == 'high':
                img.thumbnail((2000, 2000))
                if min(img.size) < 768:
                    scale = 768 / min(img.size)
                    new_size = tuple(int(x * scale) for x in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
            elif mode == 'low':
                img = img.resize((512, 512))

            buffer = BytesIO()
            img_format = img.format or _infer_image_format(attachment_filename)
            img.save(buffer, format=img_format)
            return buffer.getvalue()
    except Exception as e:
        logger.error(f"Erreur lors du redimensionnement de l'image : {e}")
        raise

def _infer_image_format(filename):
    """Déduit le format de l'image basé sur l'extension du fichier."""
    if filename:
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        format_mapping = {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.gif': 'GIF',
            '.bmp': 'BMP',
            '.tiff': 'TIFF'
        }
        return format_mapping.get(ext, 'PNG')
    return 'PNG'

def extract_text_from_message(message):
    """Extrait le texte du message."""
    content = message.get("content", "")
    if isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")]
        return ' '.join(texts)
    elif isinstance(content, str):
        return content
    return ""

def calculate_cost(usage, model='gpt-4o-mini'):
    """Calcule le coût basé sur l'utilisation des tokens."""
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)

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

    rates = model_costs.get(model, model_costs['gpt-4o-mini'])
    input_cost = input_tokens * rates['input_rate']
    output_cost = output_tokens * rates['output_rate']
    total_cost = input_cost + output_cost

    if model not in model_costs:
        logger.warning(f"Modèle inconnu '{model}'. Utilisation des tarifs par défaut pour 'gpt-4o-mini'.")

    return input_tokens, output_tokens, total_cost

async def read_text_file(attachment):
    """Lit le contenu d'un fichier texte attaché."""
    file_bytes = await attachment.read()
    return file_bytes.decode('utf-8')

async def encode_image_from_attachment(attachment, mode='high'):
    """Encode une image depuis une pièce jointe en base64 après redimensionnement."""
    image_data = await attachment.read()
    resized_image = resize_image(image_data, mode=mode, attachment_filename=attachment.filename)
    return base64.b64encode(resized_image).decode('utf-8')

# ========================
# Interaction avec OpenAI
# ========================

# Charger l'encodeur pour le modèle GPT-4o mini
encoding = tiktoken.get_encoding("o200k_base")

async def call_openai_model(model, messages, max_tokens, temperature=0.8):
    """Appelle un modèle OpenAI avec les paramètres spécifiés et gère la réponse."""
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if response and response.choices:
            reply = response.choices[0].message.content

            # Ne pas logger les réponses de 'gpt-4o-mini' et 'gpt-4o'
            if model not in ["gpt-4o-mini", "gpt-4o"]:
                logger.info(f"Réponse de {model}: {reply[:100]}...")
            
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
                _, _, total_cost = calculate_cost(usage, model=model)
                # Log avec les tokens d'entrée et de sortie
                logger.info(f"Coût de l'utilisation de {model}: ${total_cost:.4f} / Input: {usage['prompt_tokens']} / Output: {usage['completion_tokens']}")
            else:
                logger.warning(f"Informations d'utilisation non disponibles pour {model}.")
            
            return reply
    except OpenAIError as e:
        logger.error(f"Erreur lors de l'appel à l'API OpenAI avec {model}: {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'appel à l'API OpenAI avec {model}: {e}")
    
    return None

async def call_gpt4o_for_image_analysis(image_data, user_text=None, detail='high'):
    """Appelle GPT-4o pour analyser une image."""
    prompt = (
        "Tu es un expert en analyse d'images et de textes, spécialisé dans l'étude du corps humain. "
        "On te présente une image ou un texte qui pourrait contenir des informations importantes. "
        "Analyse chaque détail de manière méticuleuse. "
        "Si l'image montre un environnement sans personnage, décris minutieusement les objets, leur disposition, les couleurs, textures, formes, et tout autre élément notable. "
        "Si du texte est présent, analyse chaque mot attentivement : style, mise en page, ou tout détail subtil qui pourrait en révéler plus sur le contexte ou l'intention. "
        "Si des personnages sont présents, décris-les avec précision, dans l'ordre suivant : D'abord leur posture et expression faciale. "
        "Puis leurs vêtements dans leur intégralité. "
        "Puis leur physique, en commençant par les cheveux, visage, épaules, poitrine, ventre, taille, hanches, fesses, cuisses, jambes et pieds."
    )
    if user_text:
        prompt += f" Voici ce que l'on te décrit : \"{user_text}\"."
    
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
    
    messages = [message_to_send]
    analysis = await call_openai_model(
        model="gpt-4o",
        messages=messages,
        max_tokens=4096,
        temperature=1.0
    )
    
    if analysis:
        logger.info(f"Analyse de l'image par GPT-4o : {analysis}")
    return analysis

async def call_gpt4o_mini_with_analysis(analysis_text, user_name, user_question, has_text_flag):
    """Appelle GPT-4o Mini pour générer une réponse basée sur l'analyse de l'image."""
    system_messages = [
        {"role": "system", "content": PERSONALITY_PROMPT},
        {
            "role": "system",
            "content": f"L'analyse de l'image fournie est la suivante :\n{analysis_text}\n\n"
        }
    ]

    if has_text_flag:
        user_content = (
            f"Tu es Rika, une femme de 24 ans et membre à part entière du serveur Discord sur lequel tu te trouves. "
            f"{user_name} a posté un message contenant une image et a écrit avec : '{user_question}'. "
            "Réponds à l'utilisateur en te basant sur l'analyse, avec ta personnalité. "
            "Ne mentionne pas explicitement que l'analyse est pré-existante, fais comme si tu l'avais faite toi-même."
        )
    else:
        user_content = (
            f"Tu es Rika, une femme de 24 ans et membre à part entière du serveur Discord sur lequel tu te trouves. "
            f"{user_name} a partagé une image sans texte additionnel. "
            "Commente l'image en te basant sur l'analyse, avec ta personnalité. "
            "Ne mentionne pas que l'analyse a été fournie à l'avance, réagis comme si tu l'avais toi-même effectuée."
        )

    user_message = {"role": "user", "content": user_content}
    messages = system_messages + conversation_history + [user_message]
    
    reply = await call_openai_model(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=450,
        temperature=1.0
    )
    
    return reply

async def call_openai_api(user_text, user_name, image_data=None, detail='high'):
    """Appelle l'API OpenAI pour générer une réponse basée sur le texte et/ou l'image."""
    text = f"{user_name} dit : {user_text}"
    if image_data:
        text += " (a posté une image.)"

    message_to_send = {
        "role": "user",
        "content": [
            {"type": "text", "text": text}
        ]
    }

    if image_data:
        message_to_send["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
                "detail": detail
            }
        })

    messages = [
        {"role": "system", "content": PERSONALITY_PROMPT}
    ] + conversation_history + [message_to_send]

    reply = await call_openai_model(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=450,
        temperature=1.0
    )
    
    return reply

# ============================
# Gestion du contenu de l'Historique
# ============================

async def remove_old_image_analyses(db_connection, new_analysis=False):
    """Supprime les anciennes analyses d'images de l'historique."""
    global conversation_history, last_analysis_index, messages_since_last_analysis

    if new_analysis:
        logger.debug("Nouvelle analyse détectée. Suppression des anciennes analyses.")
        conversation_history = [
            msg for msg in conversation_history
            if not (msg.get("role") == "system" and msg.get("content", "").startswith("__IMAGE_ANALYSIS__:"))
        ]
        last_analysis_index = len(conversation_history)
        messages_since_last_analysis = 0

        # Supprimer les analyses d'images de la base de données
        try:
            cursor = db_connection.cursor()
            cursor.execute("DELETE FROM conversation_history WHERE role = 'system' AND content LIKE '__IMAGE_ANALYSIS__:%'")
            db_connection.commit()
            logger.info("Toutes les anciennes analyses d'image ont été supprimées de la base de données.")
        except Error as e:
            logger.error(f"Erreur lors de la suppression des analyses d'image: {e}")
        finally:
            cursor.close()

async def add_to_conversation_history(db_connection, new_message):
    global conversation_history, last_analysis_index, messages_since_last_analysis

    # Exclure le PERSONALITY_PROMPT de l'historique
    if new_message.get("role") == "system" and new_message.get("content") == PERSONALITY_PROMPT:
        logger.debug("PERSONALITY_PROMPT système non ajouté à l'historique.")
        return

    # Gérer les analyses d'images
    if new_message.get("role") == "system" and new_message.get("content", "").startswith("__IMAGE_ANALYSIS__:"):
        await remove_old_image_analyses(db_connection, new_analysis=True)

    # Ajouter le message à l'historique en mémoire
    conversation_history.append(new_message)
    # Sauvegarder dans la base de données
    save_message_to_db(db_connection, new_message.get("role"), new_message.get("content"))

    logger.debug(f"Message ajouté à l'historique. Taille actuelle : {len(conversation_history)}")

    # Mettre à jour les indices pour les analyses d'images
    if new_message.get("role") == "system" and new_message.get("content", "").startswith("__IMAGE_ANALYSIS__:"):
        last_analysis_index = len(conversation_history) - 1
        messages_since_last_analysis = 0
    else:
        await remove_old_image_analyses(db_connection, new_analysis=False)

    # Limiter l'historique à 150 messages
    if len(conversation_history) > 150:
        excess = len(conversation_history) - 150
        conversation_history = conversation_history[excess:]
        # Supprimer les messages les plus anciens de la base de données
        try:
            cursor = db_connection.cursor()
            cursor.execute("DELETE FROM conversation_history ORDER BY id ASC LIMIT %s", (excess,))
            db_connection.commit()
            logger.debug(f"{excess} messages les plus anciens ont été supprimés de la base de données pour maintenir l'historique à 150 messages.")
        except Error as e:
            logger.error(f"Erreur lors de la suppression des anciens messages: {e}")
        finally:
            cursor.close()

# =====================
# Gestion des Événements Discord
# =====================

@client_discord.event
async def on_ready():
    """Événement déclenché lorsque le bot est prêt."""
    logger.info(f'{BOT_NAME} connecté en tant que {client_discord.user}')

    if not conversation_history:
        logger.info("Aucun historique trouvé. L'historique commence vide.")

    # Envoyer un message de version dans le canal Discord
    channel = client_discord.get_channel(chatgpt_channel_id)
    if channel:
        try:
            embed = discord.Embed(
                title="Bot Démarré",
                description=f"🎉 {BOT_NAME} est en ligne ! Version {BOT_VERSION}",
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
    """Événement déclenché lorsqu'un message est envoyé dans un canal suivi."""
    global conversation_history, last_analysis_index, messages_since_last_analysis

    # Ignorer les messages provenant d'autres canaux ou du bot lui-même
    if message.channel.id != chatgpt_channel_id or message.author == client_discord.user:
        return

    user_text = message.content.strip()

    # Commande de réinitialisation de l'historique
    if user_text.lower() == "!reset_history":
        if not message.author.guild_permissions.administrator:
            await message.channel.send("❌ Vous n'avez pas la permission d'utiliser cette commande.")
            return

        conversation_history = []
        try:
            cursor = db_connection.cursor()
            cursor.execute("DELETE FROM conversation_history")
            db_connection.commit()
            logger.info(f"Historique des conversations réinitialisé par {message.author}.")
            await message.channel.send("✅ L'historique des conversations a été réinitialisé.")
        except Error as e:
            logger.error(f"Erreur lors de la réinitialisation de l'historique: {e}")
            await message.channel.send("❌ Une erreur est survenue lors de la réinitialisation de l'historique.")
        finally:
            cursor.close()
        return

    # Traiter les pièces jointes
    image_data = None
    file_content = None
    attachment_filename = None
    allowed_extensions = ['.txt', '.py', '.html', '.css', '.js']

    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in allowed_extensions):
                file_content = await read_text_file(attachment)
                attachment_filename = attachment.filename
                break
            elif attachment.content_type and attachment.content_type.startswith('image/'):
                image_data = await encode_image_from_attachment(attachment, mode='high')
                break

    # Traitement des images
    if image_data:
        has_user_text = has_text(user_text)
        user_text_to_use = user_text if has_user_text else None

        temp_msg = await message.channel.send(f"*{BOT_NAME} observe l'image...*")

        try:
            # Analyser l'image avec GPT-4o
            analysis = await call_gpt4o_for_image_analysis(image_data, user_text=user_text_to_use)

            if analysis:
                # Ajouter l'analyse à l'historique
                analysis_message = {
                    "role": "system",
                    "content": f"__IMAGE_ANALYSIS__:{analysis}"
                }
                await add_to_conversation_history(db_connection, analysis_message)

                # Générer une réponse basée sur l'analyse
                reply = await call_gpt4o_mini_with_analysis(analysis, message.author.name, user_text, has_user_text)
                if reply:
                    await temp_msg.delete()
                    await message.channel.send(reply)

                    # Construire et ajouter les messages à l'historique
                    user_message_text = f"{user_text} (a posté une image.)" if has_user_text else (
                        "Une image a été postée, mais elle n'est pas disponible pour analyse directe. Veuillez vous baser uniquement sur l'analyse fournie."
                    )
                    user_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{message.author.name} dit : {user_message_text}"}
                        ]
                    }
                    assistant_message = {
                        "role": "assistant",
                        "content": reply
                    }

                    await add_to_conversation_history(db_connection, user_message)
                    await add_to_conversation_history(db_connection, assistant_message)
                else:
                    await temp_msg.delete()
                    await message.channel.send("Désolé, je n'ai pas pu générer une réponse.")
            else:
                await temp_msg.delete()
                await message.channel.send("Désolé, je n'ai pas pu analyser l'image.")

        except Exception as e:
            await temp_msg.delete()
            await message.channel.send("Une erreur est survenue lors du traitement de l'image.")
            logger.error(f"Erreur lors du traitement de l'image: {e}")

        return  # Ne pas continuer le traitement après une image

    # Ajouter le contenu du fichier au texte de l'utilisateur si un fichier est présent
    if file_content:
        user_text += f"\nContenu du fichier {attachment_filename}:\n{file_content}"

    # Vérifier si le texte n'est pas vide
    if not has_text(user_text):
        return  # Ne pas appeler l'API si le texte est vide

    # Appeler l'API OpenAI pour le texte
    reply = await call_openai_api(user_text, message.author.name)
    if reply:
        await message.channel.send(reply)

        # Construire et ajouter les messages à l'historique
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{message.author.name} dit : {user_text}"}
            ]
        }
        assistant_message = {
            "role": "assistant",
            "content": reply
        }

        await add_to_conversation_history(db_connection, user_message)
        await add_to_conversation_history(db_connection, assistant_message)

# ============================
# Démarrage du Bot Discord
# ============================

if __name__ == "__main__":
    db_connection = create_db_connection()
    if db_connection:
        load_conversation_history(db_connection)
        client_discord.run(DISCORD_TOKEN)
        db_connection.close()
    else:
        logger.error("Le bot ne peut pas démarrer sans connexion à la base de données.")
