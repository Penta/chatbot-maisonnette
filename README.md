# Chatbot Discord — version 5.0 (Claude)

Bot Discord qui discute avec les membres du serveur grâce à Claude (Anthropic).

## Fonctionnalités

- **Salon dédié** : répond à tous les messages du salon défini par `CHANNEL_ID`, avec une mémoire des derniers échanges, sauvegardée entre deux redémarrages.
- **Mode mention** : dans les autres salons, répond quand on le mentionne, en lisant les messages récents pour comprendre la discussion.
- **Images** : commente une image envoyée dans le salon dédié (JPEG, PNG, GIF, WebP, une par message).
- **Fichiers texte** : lit un fichier texte joint au message (`.txt`, `.md`, `.csv`, `.json`, `.log`, sous-titres, code…, un par message, 100 Ko maximum), dans le salon dédié et en mode mention. Seul le nom du fichier est gardé en mémoire, pas son contenu.
- **PDF** : lit un PDF joint au message (texte, tableaux, schémas et pages scannées), à la place d'une image, dans le salon dédié et en mode mention. 20 pages maximum par défaut (`PDF_MAX_PAGES`, 0 pour refuser les PDF) et 20 Mo ; les PDF protégés par un mot de passe sont refusés. Chaque page coûte souvent plusieurs milliers de tokens. Seul le nom du fichier est gardé en mémoire.
- **Réponses** : quand on utilise « Répondre » sur un message, même ancien, le bot lit le message cité (et son image, si on n'en joint pas une autre).
- **Transferts** : dans le salon dédié, le bot lit un message transféré (son texte et son image).
- **Autres bots** : le bot ignore les autres bots, sauf quand ils le mentionnent ou répondent à l'un de ses messages (bouton « Répondre ») : deux bots peuvent ainsi discuter ensemble. Les messages système (arrivées, épinglages, boosts…) sont ignorés.
- **Stickers et emojis** : répond à un sticker seul par un sticker du serveur, et à un emoji personnalisé seul par un emoji du serveur (avec des mots si le serveur n'en a aucun d'utilisable). Un sticker accompagné de texte est décrit à Claude avec le message.
- **Statut Discord** : sous son pseudo, le bot regarde quelques épisodes d'un anime (top 500 d'AniList), écoute quelques morceaux (tops Deezer), joue à un jeu (toute la boutique Steam, sans les jeux pour adultes), affiche une pensée du moment, ou passe en « Ne pas déranger » ou absent avec une raison écrite par Claude. Chaque statut a sa propre durée (réglages `PRESENCE_*` du `.env`), le bot est absent la nuit, et il sait ce qu'il est en train de faire quand on lui parle. Absent, il ne répond qu'à son retour ; en « Ne pas déranger », au bout de 5 à 15 minutes. S'il change de statut en pleine conversation (message dans son salon depuis moins de 3 minutes), il prévient d'abord qu'il s'absente. Les messages reçus entre-temps reçoivent une réponse groupée dans le salon dédié, et une réponse à chaque mention ailleurs (gardés dans `waiting_messages.json`, même après un redémarrage). Les listes sont téléchargées au démarrage puis une fois par jour, et gardées dans `presence_sources.json` à côté de l'historique.
- **`/reset`** : efface la mémoire du salon (réservé aux modérateurs et administrateurs).
- **Personnalité** modifiable dans `personality_prompt.txt`, prise en compte sans redémarrer. Pour une personnalité perso, crée `personality_prompt.local.txt` et indique-le dans `PERSONALITY_FILE` du `.env` : les fichiers `*.local.txt` ne sont jamais envoyés sur GitHub.
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
bot/presence.py       statut Discord : tirage au sort et changements
bot/presence_sources.py  animes, musiques et jeux (AniList, Deezer, Steam)
bot/text_files.py     lecture des fichiers texte joints (.txt, .md, code…)
bot/pdf_files.py      vérification des PDF joints (pages, mot de passe)
bot/text_utils.py     découpage des longues réponses
```
