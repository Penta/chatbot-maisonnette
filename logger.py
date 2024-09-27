# Configurer et initialiser le système de journalisation (logging).

import logging
from config import BOT_NAME

# Configuration du format de log
log_format = '%(asctime)-13s : %(name)-15s : %(levelname)-8s : %(message)s'

# Configuration de base du logger avec un fichier de log
logging.basicConfig(
    handlers=[logging.FileHandler("./chatbot.log", 'a', 'utf-8')],
    format=log_format,
    level=logging.INFO
)

# Configuration du logger pour la console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(log_format))

# Création du logger principal
logger = logging.getLogger(BOT_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(console)

# Configuration spécifique pour httpx
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)
