# Utiliser une image de base Python
FROM python:3.12

# Définir le répertoire de travail dans le conteneur
WORKDIR /opt/chatbot

# Copier le fichier des dépendances dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Spécifier la commande pour lancer l'application
CMD ["python", "chatbot.py"]
