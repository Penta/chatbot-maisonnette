# Point d'entrée principal pour démarrer le bot Discord.

from bot import client_discord
from config import DISCORD_TOKEN

# Démarrer le bot Discord
client_discord.run(DISCORD_TOKEN)
