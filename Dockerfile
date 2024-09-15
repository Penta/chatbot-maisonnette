# Utiliser une image de base Python
FROM python:3.12

# Créer un utilisateur non root
RUN useradd -u 4050 chatbot

# Définir le répertoire de travail dans le conteneur
WORKDIR /opt/chatbot

# Copier le fichier des dépendances dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

# S'assurer que les fichiers sont accessibles à l'utilisateur non root
RUN chown -R 4050:0 /opt/chatbot && chmod -R g+rw /opt/chatbot

# Utiliser l'utilisateur non root
USER 4050

# Spécifier la commande pour lancer l'application
CMD ["python", "chatbot.py"]
