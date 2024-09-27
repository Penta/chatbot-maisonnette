# Contenir les fonctions utilitaires auxiliaires.

import re
import os
import base64
from io import BytesIO
from PIL import Image
from logger import logger
import tiktoken

def has_text(text):
    """
    Détermine si le texte fourni est non vide après suppression des espaces.
    """
    return bool(text.strip())

def resize_image(image_bytes, mode='high', attachment_filename=None):
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            original_format = img.format  # Stocker le format original

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

async def read_text_file(attachment):
    file_bytes = await attachment.read()
    return file_bytes.decode('utf-8')

async def encode_image_from_attachment(attachment, mode='high'):
    image_data = await attachment.read()
    resized_image = resize_image(image_data, mode=mode, attachment_filename=attachment.filename)
    return base64.b64encode(resized_image).decode('utf-8')
