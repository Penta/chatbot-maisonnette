# Gérer la configuration et les variables d'environnement.

import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Variables de configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
PERSONALITY_PROMPT_FILE = os.getenv('PERSONALITY_PROMPT_FILE', 'personality_prompt.txt')
CONVERSATION_HISTORY_FILE = os.getenv('CONVERSATION_HISTORY_FILE', 'conversation_history.json')
BOT_NAME = os.getenv('BOT_NAME', 'ChatBot')
BOT_VERSION = "2.5.0"

# Vérifications des configurations essentielles
if DISCORD_TOKEN is None or OPENAI_API_KEY is None or DISCORD_CHANNEL_ID is None:
    raise ValueError("Les tokens ou l'ID du canal ne sont pas définis dans les variables d'environnement.")

# Vérification de l'existence du fichier de prompt de personnalité
if not os.path.isfile(PERSONALITY_PROMPT_FILE):
    raise FileNotFoundError(f"Le fichier de prompt de personnalité '{PERSONALITY_PROMPT_FILE}' est introuvable.")

# Lecture du prompt de personnalité
with open(PERSONALITY_PROMPT_FILE, 'r', encoding='utf-8') as f:
    PERSONALITY_PROMPT = f.read().strip()

# Conversion de l'ID du canal Discord en entier
try:
    CHATGPT_CHANNEL_ID = int(DISCORD_CHANNEL_ID)
except ValueError:
    raise ValueError("L'ID du channel Discord est invalide. Assurez-vous qu'il s'agit d'un entier.")
