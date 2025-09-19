import discord
from discord.ext import commands
import requests
import json
import os
import random
import re
from dotenv import load_dotenv
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Configuration du logger
logger = logging.getLogger('discord_bot')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = RotatingFileHandler('bot.log', maxBytes=5*1024*1024, backupCount=2)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Charger les variables d'environnement
load_dotenv()

# Version du bot
VERSION = "4.6.2"

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
    MISTRAL_API_KEY = get_env_variable('MISTRAL_API_KEY')
    DISCORD_TOKEN = get_env_variable('DISCORD_TOKEN')
    CHANNEL_ID = get_env_variable('CHANNEL_ID', var_type=int)
    MAX_HISTORY_LENGTH = get_env_variable('MAX_HISTORY_LENGTH', is_critical=False, default=10, var_type=int)
    CONTEXT_MESSAGE_LIMIT = get_env_variable('CONTEXT_MESSAGE_LIMIT', is_critical=False, default=20, var_type=int)
    MAX_IMAGE_SIZE = get_env_variable('MAX_IMAGE_SIZE', is_critical=False, default=5*1024*1024, var_type=int)
    HISTORY_FILE = get_env_variable('HISTORY_FILE', is_critical=False, default="conversation_history.json")
    MISTRAL_MODEL = get_env_variable('MISTRAL_MODEL', is_critical=False, default="mistral-medium-latest")
    logger.info("Toutes les variables d'environnement critiques ont été chargées avec succès.")
except ValueError as e:
    logger.error(f"Erreur lors du chargement des variables d'environnement: {e}")
    exit(1)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def split_message(message, max_length=2000):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

class ConversationHistory:
    def __init__(self, file_path, max_length):
        self.file_path = file_path
        self.max_length = max_length
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for channel_id in data:
                        if "messages" in data[channel_id]:
                            if len(data[channel_id]["messages"]) > self.max_length:
                                data[channel_id]["messages"] = data[channel_id]["messages"][-self.max_length:]
                    return data
            except json.JSONDecodeError:
                logger.error("Erreur de lecture du fichier d'historique. Création d'un nouveau fichier.")
                return {}
        return {}

    def save_history(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def add_message(self, channel_id, message):
        if channel_id not in self.history:
            self.history[channel_id] = {"messages": []}
        self.history[channel_id]["messages"].append(message)
        if len(self.history[channel_id]["messages"]) > self.max_length:
            self.history[channel_id]["messages"] = self.history[channel_id]["messages"][-self.max_length:]
        self.save_history()

    def get_history(self, channel_id):
        if channel_id in self.history:
            return self.history[channel_id]
        else:
            self.history[channel_id] = {"messages": []}
            return self.history[channel_id]

    def reset_history(self, channel_id):
        if channel_id in self.history:
            self.history[channel_id]["messages"] = []
        else:
            self.history[channel_id] = {"messages": []}
        self.save_history()

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

history_manager = ConversationHistory(HISTORY_FILE, MAX_HISTORY_LENGTH)

def call_mistral_api(prompt, history, image_url=None, user_id=None, username=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    personality_prompt = get_personality_prompt()
    current_time = datetime.now().strftime("%d-%m-%y %H:%M")
    if image_url:
        user_content = [
            {"type": "text", "text": f"{current_time}, {username} a écrit : {prompt}" if username else f"{current_time}, a écrit : {prompt}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"
                }
            }
        ]
        user_message = {"role": "user", "content": user_content}
    else:
        user_content = [{"type": "text", "text": f"{current_time}, {username} a écrit : {prompt}" if username else f"{current_time}, a écrit : {prompt}"}]
        user_message = {"role": "user", "content": user_content}
    history["messages"].append(user_message)
    if len(history["messages"]) > MAX_HISTORY_LENGTH:
        history["messages"] = history["messages"][-MAX_HISTORY_LENGTH:]
    messages = [{"role": "system", "content": personality_prompt}]
    for msg in history["messages"]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"] if isinstance(msg["content"], list) else msg["content"]
        })
    data = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "max_tokens": 128000
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                assistant_response = response_data['choices'][0]['message']['content']
                history["messages"].append({"role": "assistant", "content": assistant_response})
                if 'usage' in response_data:
                    prompt_tokens = response_data['usage']['prompt_tokens']
                    completion_tokens = response_data['usage']['completion_tokens']
                    input_cost = (prompt_tokens / 1_000_000) * 0.4
                    output_cost = (completion_tokens / 1_000_000) * 2
                    total_cost = input_cost + output_cost
                    logger.info(f"Mistral API Call - Input Tokens: {prompt_tokens}, Output Tokens: {completion_tokens}, Cost: ${total_cost:.6f}")
                else:
                    logger.warning("La réponse de l'API ne contient pas d'informations sur les tokens.")
                if len(history["messages"]) > MAX_HISTORY_LENGTH:
                    history["messages"] = history["messages"][-MAX_HISTORY_LENGTH:]
                history_manager.save_history()
                return assistant_response
            else:
                logger.error(f"Réponse API inattendue: {response_data}")
                return "Désolé, je n'ai pas reçu de réponse valide de l'API."
        else:
            return f"Erreur API: {response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'appel API: {e}")
        return "Désolé, une erreur réseau est survenue lors de la communication avec l'API."

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.presences = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'Le bot est connecté en tant que {bot.user}')
    history_manager.history = history_manager.load_history()
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
    await bot.tree.sync()

@bot.tree.command(name="reset", description="Réinitialise l'historique de conversation")
async def reset_history_slash(interaction: discord.Interaction):
    channel_id = str(interaction.channel.id)
    history_manager.reset_history(channel_id)
    await interaction.response.send_message("L'historique de conversation a été réinitialisé.")

async def handle_stickers(message):
    if message.stickers:
        guild = message.guild
        if guild:
            stickers = guild.stickers
            if stickers:
                random_stickers = random.sample(stickers, len(stickers))
                for sticker in random_stickers:
                    try:
                        logger.info(f"Envoi du sticker: {sticker.name} (ID: {sticker.id})")
                        await message.channel.send(stickers=[sticker])
                        break
                    except discord.errors.Forbidden as e:
                        logger.error(f"Erreur lors de l'envoi du sticker: {sticker.name} (ID: {sticker.id}). Erreur: {e}")
                        continue
                else:
                    logger.error("Aucun sticker utilisable trouvé sur ce serveur.")
                    await message.channel.send("Aucun sticker utilisable trouvé sur ce serveur.")
            else:
                await message.channel.send("Aucun sticker personnalisé trouvé sur ce serveur.")
        else:
            await message.channel.send("Ce message ne provient pas d'un serveur.")
        return True
    return False

async def handle_emojis(message):
    emoji_pattern = re.compile(r'^<a?:\w+:\d+>$')
    content = message.content.strip()
    if emoji_pattern.match(content):
        guild = message.guild
        if guild and guild.emojis:
            random_emoji = random.choice(guild.emojis)
            try:
                await message.channel.send(str(random_emoji))
                return True
            except discord.errors.Forbidden as e:
                logger.error(f"Erreur lors de l'envoi de l'emoji: {random_emoji.name} (ID: {random_emoji.id}). Erreur: {e}")
                await message.channel.send("Je n'ai pas pu envoyer d'emoji en réponse.")
        else:
            await message.channel.send("Aucun emoji personnalisé trouvé sur ce serveur.")
        return True
    return False

async def handle_images(message):
    if message.attachments:
        image_count = 0
        non_image_files = []
        too_large_images = []
        max_size = MAX_IMAGE_SIZE
        for attachment in message.attachments:
            is_image = False
            if attachment.content_type and attachment.content_type.startswith('image/'):
                is_image = True
            else:
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']
                if any(attachment.filename.lower().endswith(ext) for ext in image_extensions):
                    is_image = True
            if is_image:
                image_count += 1
                if attachment.size > max_size:
                    too_large_images.append(attachment.filename)
            else:
                non_image_files.append(attachment.filename)
        if non_image_files:
            file_list = ", ".join(non_image_files)
            await message.channel.send(f"Erreur : Les fichiers suivants ne sont pas des images et ne sont pas pris en charge : {file_list}. Veuillez envoyer uniquement des images.")
            return False
        if image_count > 1:
            await message.channel.send("Erreur : Vous ne pouvez pas envoyer plus d'une image en un seul message.")
            return False
        if too_large_images:
            image_list = ", ".join(too_large_images)
            max_size_mb = max_size / (1024 * 1024)
            await message.channel.send(f"Erreur : Les images suivantes dépassent la limite de {max_size_mb} Mo : {image_list}. Veuillez envoyer des images plus petites.")
            return False
    return True

async def handle_bot_mention(message):
    context_messages = []
    async for msg in message.channel.history(limit=CONTEXT_MESSAGE_LIMIT, before=message):
        resolved_content = msg.content
        for user in msg.mentions:
            resolved_content = resolved_content.replace(f"<@{user.id}>", f"@{user.display_name}")
        author_name = msg.author.display_name
        context_messages.append(f"{author_name}: {resolved_content}")
    context_messages.reverse()
    context = "\n".join(context_messages)
    resolved_content = message.content
    for user in message.mentions:
        resolved_content = resolved_content.replace(f"<@{user.id}>", f"@{user.display_name}")
    bot_mention = f"<@{bot.user.id}>"
    if bot_mention in resolved_content:
        resolved_content = resolved_content.replace(bot_mention, "").strip()
    prompt = f"Contexte de la conversation récente:\n{context}\n\nNouveau message: {resolved_content}"
    channel_id = str(message.channel.id)
    temp_history = {
        "messages": [
            {"role": "system", "content": get_personality_prompt()},
            {"role": "user", "content": prompt}
        ]
    }
    async with message.channel.typing():
        try:
            response = call_mistral_api(
                prompt,
                temp_history,
                None,
                user_id=str(message.author.id),
                username=message.author.display_name
            )
            if len(response) > 2000:
                chunks = split_message(response)
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API: {e}")
            await message.channel.send("Désolé, une erreur est survenue lors du traitement de votre demande.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if bot.user.mentioned_in(message):
        if message.channel.id == CHANNEL_ID:
            pass
        else:
            await handle_bot_mention(message)
            return
    if message.channel.id != CHANNEL_ID:
        return
    if await handle_stickers(message):
        return
    if await handle_emojis(message):
        return
    if not await handle_images(message):
        return
    channel_id = str(message.channel.id)
    history = history_manager.get_history(channel_id)
    image_url = None
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_url = attachment.url
                break
    resolved_content = message.content
    for user in message.mentions:
        resolved_content = resolved_content.replace(f"<@{user.id}>", f"@{user.display_name}")
    prompt = resolved_content
    async with message.channel.typing():
        try:
            response = call_mistral_api(
                prompt,
                history,
                image_url,
                user_id=str(message.author.id),
                username=message.author.display_name
            )
            if len(response) > 2000:
                chunks = split_message(response)
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API: {e}")
            await message.channel.send("Désolé, une erreur est survenue lors du traitement de votre demande.")
    await bot.process_commands(message)

bot.run(DISCORD_TOKEN)
