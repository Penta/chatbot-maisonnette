# Gérer le chargement et la sauvegarde de l'historique des conversations.

import os
import json
from logger import logger
from config import CONVERSATION_HISTORY_FILE, PERSONALITY_PROMPT

# Variable globale pour l'historique des conversations
conversation_history = []

# Constantes pour les limites de l'historique
STANDARD_HISTORY_LIMIT = 150
IMAGE_ANALYSIS_HISTORY_LIMIT = 15
IMAGE_ANALYSIS_MARKER = "__IMAGE_ANALYSIS__:"

def load_conversation_history():
    global conversation_history
    if os.path.isfile(CONVERSATION_HISTORY_FILE):
        try:
            with open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
                # Exclure le PERSONALITY_PROMPT
                filtered_history = [
                    msg for msg in loaded_history
                    if not (msg.get("role") == "system" and msg.get("content") == PERSONALITY_PROMPT)
                ]
                # Modifier la liste en place
                conversation_history.clear()
                conversation_history.extend(filtered_history)
            logger.info(f"Historique chargé depuis {CONVERSATION_HISTORY_FILE} : {len(loaded_history)} messages, {len(conversation_history)} après filtrage.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique : {e}")
            conversation_history = []
    else:
        logger.info(f"Aucun fichier d'historique trouvé. Un nouveau fichier sera créé à {CONVERSATION_HISTORY_FILE}")

def save_conversation_history():
    try:
        with open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'historique : {e}")

async def add_to_conversation_history(new_message):
    global conversation_history
    # Ne pas ajouter le PERSONALITY_PROMPT à l'historique
    if new_message.get("role") == "system" and new_message.get("content") == PERSONALITY_PROMPT:
        logger.debug("PERSONALITY_PROMPT système non ajouté à l'historique.")
        return

    is_image_analysis = new_message.get("role") == "system" and new_message.get("content", "").startswith(IMAGE_ANALYSIS_MARKER)

    if is_image_analysis:
    	# Supprimer toutes les analyses d'images précédentes
    	before_count = len(conversation_history)
    	conversation_history = [
    		msg for msg in conversation_history
    		if not (msg.get("role") == "system" and msg.get("content", "").startswith(IMAGE_ANALYSIS_MARKER))
    	]
    	after_count = len(conversation_history)
    	if before_count != after_count:
    		logger.info("Les analyses d'images précédentes ont été supprimées avant d'ajouter la nouvelle analyse.")

    # Ajouter le nouveau message
    conversation_history.append(new_message)
    logger.debug(f"Message ajouté à l'historique. Taille actuelle : {len(conversation_history)}")

    # Si le message ajouté est une analyse d'image, rien de plus à faire
    if is_image_analysis:
    	pass
    else:
    	# Vérifier et supprimer les analyses d'images dépassées
    	# Parcourir l'historique pour trouver les analyses d'images
    	indices_to_remove = []
    	for i, msg in enumerate(conversation_history):
    		if msg.get("role") == "system" and msg.get("content", "").startswith(IMAGE_ANALYSIS_MARKER):
    			# Calculer le nombre de messages après cette analyse
    			messages_after = len(conversation_history) - i - 1
    			if messages_after >= IMAGE_ANALYSIS_HISTORY_LIMIT:
    				indices_to_remove.append(i)

    	# Supprimer les analyses d'images identifiées
    	for i in reversed(indices_to_remove):
    		del conversation_history[i]
    		logger.info(f"Analyse d'image à l'index {i} supprimée après {IMAGE_ANALYSIS_HISTORY_LIMIT} nouveaux messages.")

    # Limiter l'historique à STANDARD_HISTORY_LIMIT messages
    if len(conversation_history) > STANDARD_HISTORY_LIMIT:
        excess_messages = len(conversation_history) - STANDARD_HISTORY_LIMIT
        conversation_history = conversation_history[excess_messages:]
        logger.debug(f"{excess_messages} messages les plus anciens ont été supprimés pour maintenir l'historique à {STANDARD_HISTORY_LIMIT} messages.")

    save_conversation_history()