# Chatbot Discord — version 5.0 (Claude)

Bot Discord qui discute avec les membres du serveur grâce à Claude (Anthropic).

## Fonctionnalités

- **Salon dédié** : répond à tous les messages du salon défini par `CHANNEL_ID`, avec une mémoire des derniers échanges, sauvegardée entre deux redémarrages.
- **Mode mention** : dans les autres salons, répond quand on le mentionne, en lisant les messages récents pour comprendre la discussion.
- **Images** : commente une image envoyée dans le salon dédié (JPEG, PNG, GIF, WebP, une par message).
- **Réponses** : quand on utilise « Répondre » sur un message, même ancien, le bot lit le message cité (et son image, si on n'en joint pas une autre).
- **Stickers et emojis** : répond à un sticker par un sticker du serveur, et à un emoji personnalisé seul par un emoji du serveur.
- **`/reset`** : efface la mémoire du salon (réservé aux modérateurs et administrateurs).
- **Personnalité** modifiable dans `personality_prompt.txt`, prise en compte sans redémarrer.
- **Logs** dans `bot.log`, avec le coût estimé de chaque appel.

## Installation

Il faut **Python 3.10 ou plus récent**.

1. Ouvre un terminal dans ce dossier et crée un environnement virtuel :
   ```
   python -m venv .venv
   ```
   Puis active-le : `.venv\Scripts\activate` sous Windows, `source .venv/bin/activate` sous macOS et Linux.

2. Installe les dépendances :
   ```
   pip install -r requirements.txt
   ```

3. Copie `.env.example` en `.env` et remplis au minimum `DISCORD_TOKEN`, `ANTHROPIC_API_KEY` et `CHANNEL_ID`.
   La clé Anthropic se crée sur la console : https://platform.claude.com (rubrique API Keys). L'API est facturée à l'usage, séparément de l'abonnement Claude Pro.

4. Dans le portail développeur Discord (https://discord.com/developers/applications), onglet **Bot**, vérifie que **Message Content Intent** est activé. L'intent « Presence » n'est plus nécessaire, tu peux le désactiver.

5. Lance le bot :
   ```
   python main.py
   ```

## Permissions Discord nécessaires

Voir les salons, Envoyer des messages, Lire l'historique des messages. Rien d'autre n'est nécessaire : le bot utilise uniquement les emojis et stickers du serveur.

## Migration depuis la version 4 (Mistral)

- Garde ton ancien `personality_prompt.txt` : il fonctionne tel quel.
- Ton ancien `conversation_history.json` est converti automatiquement au premier lancement (les images sont remplacées par « [image jointe] »).
- Dans le `.env`, remplace `MISTRAL_API_KEY` par `ANTHROPIC_API_KEY` et `MISTRAL_MODEL` par `CLAUDE_MODEL`.
- `MAX_IMAGE_SIZE` doit être abaissé à 3750000 (limite de l'API d'Anthropic).

## Organisation du code

```
main.py               lancement et logs
bot/config.py         lecture et validation du .env
bot/claude_client.py  appels à l'API de Claude
bot/history.py        mémoire des conversations
bot/discord_bot.py    événements Discord, réactions, /reset
bot/text_utils.py     découpage des longues réponses
```
