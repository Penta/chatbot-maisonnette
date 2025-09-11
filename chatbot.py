import discord
from discord.ext import commands
import requests
import json
import os
import random
import re
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Configuration du logger
logger = logging.getLogger('discord_bot')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Créer un gestionnaire de fichier avec rotation
file_handler = RotatingFileHandler('bot.log', maxBytes=5*1024*1024, backupCount=2)  # 5 Mo par fichier, garder 2 sauvegardes
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Optionnel : ajouter un gestionnaire de console pour afficher les logs dans la console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Charger les variables d'environnement
load_dotenv()

# Version du bot
VERSION = "4.4.0"  # Modifiable selon la version actuelle

# Récupérer les variables d'environnement avec validation
def get_env_variable(var_name, is_critical=True, default=None, var_type=str):
    value = os.getenv(var_name)
    if value is None:
        if is_critical:
            logger.error(f"Variable d'environnement critique manquante: {var_name}")
            if default is not None:
                logger.warning(f"Utilisation de la valeur par défaut pour {var_name}")
                return default
            else:
                raise ValueError(f"La variable d'environnement {var_name} est requise mais non définie.")
        else:
            logger.warning(f"Variable d'environnement non critique manquante: {var_name}. Utilisation de la valeur par défaut: {default}")
            return default
    if var_type == int:
        try:
            return int(value)
        except ValueError:
            logger.error(f"La variable d'environnement {var_name} doit être un entier. Valeur actuelle: {value}")
            if default is not None:
                return default
            else:
                raise ValueError(f"La variable d'environnement {var_name} doit être un entier.")
    return value

try:
    # Variables d'environnement critiques
    MISTRAL_API_KEY = get_env_variable('MISTRAL_API_KEY')
    DISCORD_TOKEN = get_env_variable('DISCORD_TOKEN')
    CHANNEL_ID = get_env_variable('CHANNEL_ID', var_type=int)

    # Variables d'environnement non critiques avec valeurs par défaut
    MAX_HISTORY_LENGTH = get_env_variable('MAX_HISTORY_LENGTH', is_critical=False, default=10, var_type=int)
    HISTORY_FILE = get_env_variable('HISTORY_FILE', is_critical=False, default="conversation_history.json")

    logger.info("Toutes les variables d'environnement critiques ont été chargées avec succès.")
except ValueError as e:
    logger.error(f"Erreur lors du chargement des variables d'environnement: {e}")
    # Si une variable critique est manquante, le bot ne peut pas fonctionner correctement.
    # Il est donc préférable de quitter le programme avec un code d'erreur.
    exit(1)

# Endpoint API Mistral
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def load_history():
    """Charge l'historique depuis un fichier JSON."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Vérifier et limiter la taille de chaque historique
                for channel_id in data:
                    if "messages" in data[channel_id]:
                        if len(data[channel_id]["messages"]) > MAX_HISTORY_LENGTH:
                            data[channel_id]["messages"] = data[channel_id]["messages"][-MAX_HISTORY_LENGTH:]
                return data
            except json.JSONDecodeError:
                logger.error("Erreur de lecture du fichier d'historique. Création d'un nouveau fichier.")
                return {}
    return {}

def save_history(history):
    """Sauvegarde l'historique dans un fichier JSON."""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def get_personality_prompt():
    try:
        with open('personality_prompt.txt', 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error("Le fichier personality_prompt.txt n'a pas été trouvé. Utilisation d'un prompt par défaut.")
        return """Tu es un assistant utile et poli qui peut analyser des images.
        Quand on te montre une image, décris-la et donne ton avis si on te le demande.
        Réponds toujours en français avec un ton naturel et amical.
        Lorsque tu analyses une image, décris d'abord ce que tu vois en détail,
        puis réponds à la question si elle est posée. Utilise un langage clair et accessible."""

# Charger l'historique au démarrage
conversation_history = load_history()

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.presences = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'Le bot est connecté en tant que {bot.user}')
    global conversation_history
    conversation_history = load_history()
    channel = bot.get_channel(CHANNEL_ID)
    if channel is not None:
        guild = channel.guild
        if guild is not None:
            bot_member = guild.me
            bot_nickname = bot_member.display_name
        else:
            bot_nickname = bot.user.name
        embed = discord.Embed(
            title="Bot en ligne",
            description=f"{bot_nickname} est désormais en ligne. Version {VERSION}.",
            color=discord.Color.green()
        )
        await channel.send(embed=embed)
    await bot.tree.sync()  # Synchroniser les commandes slash

def call_mistral_api(prompt, history, image_url=None, user_id=None, username=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    personality_prompt = get_personality_prompt()
    # Vérifier si la structure messages existe
    if "messages" not in history:
        history["messages"] = []
    # Création du message utilisateur selon qu'il y a une image ou non
    if image_url:
        # Format multimodal pour les messages avec image
        user_content = [
            {"type": "text", "text": f"{username}: {prompt}" if username else prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"  # Demander une analyse détaillée de l'image
                }
            }
        ]
        user_message = {
            "role": "user",
            "content": user_content
        }
    else:
        # Format standard pour les messages texte seulement
        user_content = f"{username}: {prompt}" if username else prompt
        user_message = {"role": "user", "content": user_content}
    # Ajouter le message utilisateur à l'historique
    history["messages"].append(user_message)
    # Limiter l'historique à MAX_HISTORY_LENGTH messages
    if len(history["messages"]) > MAX_HISTORY_LENGTH:
        history["messages"] = history["messages"][-MAX_HISTORY_LENGTH:]
    # Préparer les messages pour l'API
    messages = []
    # Ajouter le message système en premier
    messages.append({"role": "system", "content": personality_prompt})
    # Ajouter l'historique des messages (en gardant le format)
    for msg in history["messages"]:
        if isinstance(msg["content"], list):  # C'est un message multimodal
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        else:  # C'est un message texte standard
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    data = {
        "model": "mistral-medium-2508",
        "messages": messages,
        "max_tokens": 1000
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP
        if response.status_code == 200:
            response_data = response.json()
            # Vérifier si la réponse contient bien le champ attendu
            if 'choices' in response_data and len(response_data['choices']) > 0:
                assistant_response = response_data['choices'][0]['message']['content']
                # Ajouter la réponse de l'assistant à l'historique
                history["messages"].append({"role": "assistant", "content": assistant_response})
                # Limiter à nouveau après avoir ajouté la réponse
                if len(history["messages"]) > MAX_HISTORY_LENGTH:
                    history["messages"] = history["messages"][-MAX_HISTORY_LENGTH:]
                # Sauvegarder l'historique après chaque modification
                save_history(conversation_history)
                return assistant_response
            else:
                logger.error(f"Réponse API inattendue: {response_data}")
                return "Désolé, je n'ai pas reçu de réponse valide de l'API."
        else:
            return f"Erreur API: {response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'appel API: {e}")
        return "Désolé, une erreur réseau est survenue lors de la communication avec l'API."

@bot.tree.command(name="reset", description="Réinitialise l'historique de conversation")
async def reset_history_slash(interaction: discord.Interaction):
    channel_id = str(interaction.channel.id)
    if channel_id in conversation_history:
        # Conserver le même ID de conversation mais vider les messages
        if "messages" in conversation_history[channel_id]:
            conversation_history[channel_id]["messages"] = []
        else:
            conversation_history[channel_id] = {
                "conversation_id": conversation_history[channel_id].get("conversation_id", str(len(conversation_history) + 1)),
                "messages": []
            }
        save_history(conversation_history)
        await interaction.response.send_message("L'historique de conversation a été réinitialisé.")
    else:
        conversation_id = str(len(conversation_history) + 1)
        conversation_history[channel_id] = {
            "conversation_id": conversation_id,
            "messages": []
        }
        save_history(conversation_history)
        await interaction.response.send_message("Aucun historique de conversation trouvé pour ce channel. Créé un nouvel historique.")

@bot.event
async def on_message(message):
    # Ignorer les messages du bot lui-même
    if message.author == bot.user:
        return

    # Vérifier si le message provient du channel spécifique
    if message.channel.id != CHANNEL_ID:
        return

    # Vérifier si le message contient des stickers
    if message.stickers:
        # Obtenir le serveur (guild) du message
        guild = message.guild
        if guild:
            # Obtenir la liste des stickers personnalisés du serveur
            stickers = guild.stickers
            if stickers:
                # Mélanger la liste des stickers pour essayer dans un ordre aléatoire
                random_stickers = random.sample(stickers, len(stickers))
                for sticker in random_stickers:
                    try:
                        logger.info(f"Envoi du sticker: {sticker.name} (ID: {sticker.id})")
                        await message.channel.send(stickers=[sticker])
                        break  # Si ça marche, on sort de la boucle
                    except discord.errors.Forbidden as e:
                        logger.error(f"Erreur lors de l'envoi du sticker: {sticker.name} (ID: {sticker.id}). Erreur: {e}")
                        continue
                else:
                    # Si aucun sticker n'a pu être envoyé
                    logger.error("Aucun sticker utilisable trouvé sur ce serveur.")
                    await message.channel.send("Aucun sticker utilisable trouvé sur ce serveur.")
            else:
                await message.channel.send("Aucun sticker personnalisé trouvé sur ce serveur.")
        else:
            await message.channel.send("Ce message ne provient pas d'un serveur.")
        return

    # Vérifier si le message contient uniquement un emoji personnalisé
    emoji_pattern = re.compile(r'^<a?:\w+:\d+>$')
    content = message.content.strip()
    if emoji_pattern.match(content):
        guild = message.guild
        if guild and guild.emojis:
            # Choisir un emoji aléatoire parmi ceux du serveur
            random_emoji = random.choice(guild.emojis)
            try:
                await message.channel.send(str(random_emoji))
                return  # Retourner pour éviter que le reste du code ne s'exécute
            except discord.errors.Forbidden as e:
                logger.error(f"Erreur lors de l'envoi de l'emoji: {random_emoji.name} (ID: {random_emoji.id}). Erreur: {e}")
                await message.channel.send("Je n'ai pas pu envoyer d'emoji en réponse.")
        else:
            await message.channel.send("Aucun emoji personnalisé trouvé sur ce serveur.")
        return

    # Résolution des mentions dans le message
    resolved_content = message.content
    for user in message.mentions:
        # Remplacer chaque mention par le nom d'utilisateur
        resolved_content = resolved_content.replace(f"<@{user.id}>", f"@{user.display_name}")

    # Vérifier les pièces jointes pour les images et autres fichiers
    if message.attachments:
        image_count = 0
        non_image_files = []
        too_large_images = []
        max_size = 5 * 1024 * 1024  # 5 Mo en octets
        for attachment in message.attachments:
            is_image = False
            # Vérifier le content_type si disponible
            if attachment.content_type and attachment.content_type.startswith('image/'):
                is_image = True
            else:
                # Si content_type n'est pas disponible ou pas une image, vérifier l'extension
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']
                if any(attachment.filename.lower().endswith(ext) for ext in image_extensions):
                    is_image = True

            if is_image:
                image_count += 1
                if attachment.size > max_size:
                    too_large_images.append(attachment.filename)
            else:
                non_image_files.append(attachment.filename)

        # Vérifier la présence de fichiers non-image
        if non_image_files:
            file_list = ", ".join(non_image_files)
            await message.channel.send(f"Erreur : Les fichiers suivants ne sont pas des images et ne sont pas pris en charge : {file_list}. Veuillez envoyer uniquement des images.")
            return

        # Vérifier le nombre d'images
        if image_count > 1:
            await message.channel.send("Erreur : Vous ne pouvez pas envoyer plus d'une image en un seul message.")
            return

        # Vérifier la taille des images
        if too_large_images:
            image_list = ", ".join(too_large_images)
            await message.channel.send(f"Erreur : Les images suivantes dépassent la limite de 5 Mo : {image_list}. Veuillez envoyer des images plus petites.")
            return

    # Récupérer ou initialiser l'historique pour ce channel
    channel_id = str(message.channel.id)
    global conversation_history
    conversation_history = load_history()
    if channel_id not in conversation_history:
        conversation_id = str(len(conversation_history) + 1)
        conversation_history[channel_id] = {
            "conversation_id": conversation_id,
            "messages": []
        }
    if "messages" not in conversation_history[channel_id]:
        conversation_history[channel_id]["messages"] = []

    # Traitement des images dans le message
    image_url = None
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_url = attachment.url
                break

    # Utiliser le contenu résolu (avec les mentions remplacées)
    prompt = resolved_content
    # Indiquer que le bot est en train de taper
    async with message.channel.typing():
        try:
            # Appeler l'API Mistral avec l'historique de conversation, l'image si présente, et les infos de l'utilisateur
            response = call_mistral_api(
                prompt,
                conversation_history[channel_id],
                image_url,
                user_id=str(message.author.id),
                username=message.author.display_name
            )
            await message.channel.send(response)
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API: {e}")
            await message.channel.send("Désolé, une erreur est survenue lors du traitement de votre demande.")
    # Assurer que les autres gestionnaires d'événements reçoivent également le message
    await bot.process_commands(message)

bot.run(DISCORD_TOKEN)
