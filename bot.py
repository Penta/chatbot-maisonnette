# Définir le client Discord et les gestionnaires d'événements.

import discord
from discord.ext import commands
from config import CHATGPT_CHANNEL_ID, BOT_NAME, BOT_VERSION
from logger import logger
from history import load_conversation_history, conversation_history, add_to_conversation_history, save_conversation_history
from openai_client import call_gpt4o_for_image_analysis, call_gpt4o_mini_with_analysis, call_openai_api
from utils import has_text, read_text_file, encode_image_from_attachment
from history import IMAGE_ANALYSIS_MARKER
import json

class MyDiscordClient(discord.Client):
    def __init__(self, intents):
        super().__init__(intents=intents)
        self.last_analysis_index = None
        self.messages_since_last_analysis = 0

    async def close(self):
        from openai_client import openai_client
        if openai_client is not None:
            await openai_client.close()
        await super().close()

# Initialiser les intents
intents = discord.Intents.default()
intents.message_content = True  # Activer l'intent pour les contenus de message

# Initialiser le client Discord
client_discord = MyDiscordClient(intents=intents)

# Charger l'historique au démarrage
load_conversation_history()

@client_discord.event
async def on_ready():
    logger.info(f'{BOT_NAME} connecté en tant que {client_discord.user}')

    if not conversation_history:
        logger.info("Aucun historique trouvé. L'historique commence vide.")

    try:
        # Utiliser fetch_channel au lieu de get_channel
        channel = await client_discord.fetch_channel(CHATGPT_CHANNEL_ID)
        if channel:
            embed = discord.Embed(
                title="Bot Démarré",
                description=f"🎉 {BOT_NAME} est en ligne ! Version {BOT_VERSION}",
                color=0x00ff00  # Vert
            )
            await channel.send(embed=embed)
            logger.info(f"Message de connexion envoyé dans le canal ID {CHATGPT_CHANNEL_ID}")
    except discord.NotFound:
        logger.error(f"Canal avec ID {CHATGPT_CHANNEL_ID} non trouvé.")
    except discord.Forbidden:
        logger.error(f"Permissions insuffisantes pour envoyer des messages dans le canal ID {CHATGPT_CHANNEL_ID}.")
    except discord.HTTPException as e:
        logger.error(f"Erreur lors de l'envoi du message de connexion : {e}")

@client_discord.event
async def on_message(message):
    global conversation_history

    # Vérifier si le message provient du canal autorisé
    if message.channel.id != CHATGPT_CHANNEL_ID:
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

        conversation_history.clear()
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

        # Étape 1 : Envoyer un message temporaire indiquant que l'image est en cours d'analyse
        temp_msg = await message.channel.send(f"*{BOT_NAME} observe l'image...*")

        try:
            # Étape 2 : GPT-4o analyse l'image, potentiellement guidée par le texte de l'utilisateur
            analysis = await call_gpt4o_for_image_analysis(image_data, user_text=user_text_to_use)

            if analysis:
                # Ajouter l'analyse à l'historique avant de réagir avec GPT-4o Mini
                analysis_message = {
                    "role": "system",
                    "content": f"{IMAGE_ANALYSIS_MARKER}{analysis}"
                }
                await add_to_conversation_history(analysis_message)

                # Étape 3 : GPT-4o Mini réagit à la question et à l'analyse
                reply = await call_gpt4o_mini_with_analysis(
                    analysis,
                    message.author.name,
                    user_text,
                    has_text=has_user_text,
                    conversation_history=conversation_history
                )
                if reply:
                    # Étape 4 : Supprimer le message temporaire
                    await temp_msg.delete()

                    # Étape 5 : Envoyer la réponse finale
                    await message.channel.send(reply)

                    # Ajouter le message utilisateur à l'historique
                    if has_user_text:
                        user_message_content = f"{user_text} (a posté une image.)"
                    else:
                        user_message_content = (
                            "Une image a été postée, mais elle n'est pas disponible pour analyse directe. "
                            "Veuillez vous baser uniquement sur l'analyse fournie."
                        )

                    user_message = {
                        "role": "user",
                        "content": user_message_content
                    }

                    # Ajouter le message utilisateur à l'historique
                    await add_to_conversation_history(user_message)

                    # Ajouter le message assistant à l'historique
                    assistant_message = {
                        "role": "assistant",
                        "content": reply
                    }
                    await add_to_conversation_history(assistant_message)
                else:
                    # Étape 4 : Supprimer le message temporaire en cas d'échec de génération de réponse
                    await temp_msg.delete()
                    await message.channel.send("Désolé, je n'ai pas pu générer une réponse.")
            else:
                # Étape 4 : Supprimer le message temporaire en cas d'échec d'analyse
                await temp_msg.delete()
                await message.channel.send("Désolé, je n'ai pas pu analyser l'image.")

        except Exception as e:
            # Étape 4 : Supprimer le message temporaire en cas d'erreur
            await temp_msg.delete()
            await message.channel.send("Une erreur est survenue lors du traitement de l'image.")
            logger.error(f"Error during image processing: {e}")

        # Après traitement de l'image, ne pas continuer
        return

    # Ajouter le contenu du fichier à la requête si présent
    if file_content:
        user_text += f"\nContenu du fichier {attachment_filename}:\n{file_content}"

    # Vérifier si le texte n'est pas vide après ajout du contenu du fichier
    if not has_text(user_text):
        return  # Ne pas appeler l'API si le texte est vide

    # Appeler l'API OpenAI
    result = await call_openai_api(user_text, message.author.name, conversation_history, image_data)
    if result:
        reply = result.choices[0].message.content
        await message.channel.send(reply)

        # Ajouter le message utilisateur à l'historique
        user_message = {
            "role": "user",
            "content": user_text
        }
        await add_to_conversation_history(user_message)

        # Ajouter le message assistant à l'historique
        assistant_message = {
            "role": "assistant",
            "content": reply
        }
        await add_to_conversation_history(assistant_message)
