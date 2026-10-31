# Chatbot Discord (Claude)

Un bot Discord qui se comporte comme un membre du serveur : il discute dans un salon qui lui est réservé, répond quand on le mentionne ailleurs, et a une vraie présence sur le serveur (il regarde des animes, écoute de la musique, s'absente…). Il est propulsé par Claude, le modèle d'IA d'Anthropic, et sa personnalité se règle dans un simple fichier texte.

Le bot parle français. Sa façon de parler, son caractère et son nom sont ceux que tu lui donnes dans sa personnalité.

## Fonctionnalités

### Conversation

- **Salon dédié** : le bot répond à tous les messages du salon choisi (`CHANNEL_ID`). Il se souvient des derniers échanges, même après un redémarrage. Les messages envoyés pendant qu'il écrit reçoivent une seule réponse groupée, comme le ferait quelqu'un qui lit plusieurs messages d'un coup.
- **Mentions ailleurs** : dans les autres salons, il répond quand on le mentionne, en lisant les derniers messages pour comprendre la discussion (sans mémoire d'un appel à l'autre).
- **Messages cités et transférés** : avec « Répondre », il lit le message cité, même ancien, et son image. Il lit aussi les messages transférés dans son salon.
- **Messages modifiés ou supprimés** : sa mémoire suit. Un message supprimé est oublié, un message modifié prend son nouveau texte.
- **Recherche web** (en option, `WEB_SEARCH=on`) : il peut chercher sur Internet et lire un lien partagé quand la question porte sur l'actualité ou des informations précises.
- **Autres bots** : il les ignore, sauf quand ils le mentionnent ou répondent à l'un de ses messages : deux bots peuvent ainsi discuter ensemble.

### Pièces jointes

Dans son salon comme dans les mentions, le bot lit une image ou un PDF, et un fichier texte, par message :

- **Images** (JPEG, PNG, GIF, WebP, BMP, TIFF) : réduites et converties automatiquement si besoin.
- **PDF** : texte, tableaux, schémas et pages scannées, 20 pages (`PDF_MAX_PAGES`) et 20 Mo au plus. Les PDF protégés par un mot de passe sont refusés.
- **Fichiers texte** (`.txt`, `.md`, `.csv`, `.json`, `.log`, sous-titres, configuration, HTML, code…), 100 Ko au plus.

Seul le nom du fichier reste ensuite en mémoire : le bot se souvient qu'il l'a lu, sans en repayer le contenu à chaque message.

### Vie sur le serveur

- **Statut Discord** (`PRESENCE=on`) : sous son pseudo, le bot regarde quelques épisodes d'un anime (top 500 d'AniList), écoute quelques morceaux (tops Deezer), joue à un jeu de la boutique Steam, affiche une pensée du moment, ou passe absent ou en « Ne pas déranger » avec une raison qu'il écrit lui-même. Il dort la nuit (`PRESENCE_NIGHT`) et sait ce qu'il est en train de faire quand on lui en parle. Après un redémarrage, il reprend son statut là où il en serait s'il ne s'était pas arrêté.
- **Absences** : absent, il ne répond qu'à son retour ; en « Ne pas déranger », au bout de 5 à 15 minutes. S'il s'en va en pleine conversation, il prévient d'abord son salon. Les messages en attente sont gardés même après un redémarrage.
- **Emojis du serveur** : le bot connaît automatiquement tous les emojis du serveur, animés compris, et peut les utiliser dans ses réponses. Un emoji ajouté ou retiré est pris en compte tout de suite.
- **Stickers et emojis seuls** : à un sticker seul, il répond par un sticker du serveur ; à un emoji seul, par un emoji du serveur.

### Modération

- **`/reset`** : efface la mémoire du salon. Réservé aux modérateurs (« Gérer les messages ») et aux administrateurs.
- Le bot ne peut jamais pinger `@everyone`, `@here`, un rôle ou un membre.

## Avant de commencer

Il te faut :

- **Python 3.10 ou plus récent** (ou Docker, voir plus bas) ;
- **une application Discord** avec un bot (voir l'étape suivante) ;
- **une clé API Anthropic** : elle se crée sur https://platform.claude.com, rubrique API Keys. L'API est payante à l'usage, séparément de l'abonnement Claude Pro (voir [Coûts](#coûts-et-logs)) ;
- en option, **une clé API Steam** (gratuite, https://steamcommunity.com/dev/apikey) pour le statut « joue à ». Sans elle, le bot ne joue simplement à rien.

## Créer le bot sur Discord

1. Sur https://discord.com/developers/applications, crée une application (**New Application**) et donne-lui le nom du bot.
2. Onglet **Bot** :
   - clique sur **Reset Token** et copie le token : c'est le `DISCORD_TOKEN` (à garder secret) ;
   - dans **Privileged Gateway Intents**, active **Message Content Intent**. Les intents « Presence » et « Server Members » ne servent pas.
3. Onglet **OAuth2 > URL Generator** :
   - coche les scopes **bot** et **applications.commands** (ce dernier pour `/reset`) ;
   - dans les permissions, coche **View Channels**, **Send Messages** et **Read Message History** ;
   - ouvre l'adresse générée et invite le bot sur ton serveur.
4. Dans Discord, active le mode développeur (**Paramètres utilisateur > Avancés > Mode développeur**), puis fais un clic droit sur le salon du bot > **Copier l'identifiant du salon** : c'est le `CHANNEL_ID`.

Le bot n'a besoin d'aucune autre permission : il n'utilise que les emojis et stickers du serveur.

## Installation

1. Ouvre un terminal dans le dossier du bot et crée un environnement virtuel :
   ```
   python -m venv .venv
   ```
   Puis active-le : `.venv\Scripts\activate` sous Windows, `source .venv/bin/activate` sous macOS et Linux.

2. Installe les dépendances :
   ```
   pip install -r requirements.txt
   ```

3. Copie `.env.exemple` en `.env`, puis remplis au minimum `DISCORD_TOKEN`, `ANTHROPIC_API_KEY` et `CHANNEL_ID`. Ne partage jamais ton `.env` : il contient tes clés.

4. Lance le bot :
   ```
   python main.py
   ```
   Il annonce son arrivée dans son salon avec sa version. En cas d'erreur de configuration (variable manquante, token invalide, intent non activé), il s'arrête en expliquant quoi corriger.

## Configuration

Tous les réglages se trouvent dans le `.env`, et chacun est expliqué dans `.env.exemple`. Les principaux, avec la valeur proposée dans `.env.exemple` :

| Réglage | Rôle | Valeur proposée |
| --- | --- | --- |
| `CLAUDE_MODEL` | Modèle utilisé : `claude-sonnet-5-5` (recommandé), `claude-haiku-5-5` (bien moins cher) ou `claude-opus-5-5` (plus capable, plus cher) | `claude-sonnet-5-5` |
| `THINKING` | Réflexion avant de répondre : `on` (plus réfléchi, plus lent et plus cher) ou `off` | `on` |
| `EFFORT` | Niveau d'effort : `low`, `medium`, `high`, `xhigh` ou `max` ; vide = valeur du modèle | `medium` |
| `MAX_TOKENS` | Longueur maximale d'une réponse (réflexion comprise), en tokens | `3000` |
| `MAX_HISTORY_LENGTH` | Messages gardés en mémoire dans le salon dédié (jusqu'à la moitié de plus avant d'y revenir d'un coup, ce qui garde la mémoire en cache) | `10` |
| `CONTEXT_MESSAGE_LIMIT` | Messages lus autour d'une mention | `20` |
| `WEB_SEARCH` | Recherche web et lecture de liens (`on` ou `off`) ; prévois alors un `API_TIMEOUT` d'au moins 90 secondes | `off` |
| `PRESENCE` | Statut Discord et absences (`on` ou `off`) ; les `PRESENCE_*` règlent les chances de chaque statut et la nuit | `on` |
| `PDF_MAX_PAGES` | Pages maximales d'un PDF (0 = PDF refusés) | `20` |
| `PERSONALITY_FILE` | Fichier de la personnalité | `personality_prompt.txt` |
| `TIMEZONE` | Fuseau horaire de l'heure donnée à Claude | `Europe/Paris` |

Un changement du `.env` demande de redémarrer le bot.

## Personnalité

La personnalité est le texte donné à Claude avant chaque conversation : qui est le bot, son caractère, sa façon d'écrire. Elle se trouve dans `personality_prompt.txt` et peut être modifiée pendant que le bot tourne : elle est relue automatiquement.

Pour une personnalité qui ne doit pas être publiée sur GitHub, crée `personality_prompt.local.txt` et indique-le dans `PERSONALITY_FILE` : les fichiers `*.local.txt` ne sont jamais envoyés sur GitHub ni copiés dans l'image Docker.

Pas besoin d'y recopier la liste des emojis du serveur : le bot la donne lui-même à Claude. Il suffit de dire quand en mettre, par exemple « Tu termines chaque message par un seul emoji du serveur, directement après le dernier mot ».

## Docker

Le `Dockerfile` construit une image prête à lancer. À chaque push sur `main`, GitHub Actions la construit et la publie sur Docker Hub (`<compte>/chatbot-maisonnette`, avec les secrets `DOCKER_USERNAME` et `DOCKER_PASSWORD` du dépôt).

L'image ne contient ni `.env`, ni personnalité `*.local.txt`, ni historique. Les réglages se passent en variables d'environnement, et les fichiers à garder entre deux redémarrages doivent être sur un volume, par exemple :

```
docker run -d --name chatbot \
  --env-file .env \
  -e HISTORY_FILE=/data/conversation_history.json \
  -e LOG_FILE=/data/bot.log \
  -v chatbot-data:/data \
  <compte>/chatbot-maisonnette:latest
```

Pour une personnalité personnelle, monte le fichier dans le conteneur et indique son chemin dans `PERSONALITY_FILE`.

## Coûts et logs

Chaque message envoyé à Claude est facturé selon le nombre de tokens lus (personnalité, mémoire, message) et écrits (la réponse). Pour limiter la facture :

- la personnalité et la mémoire sont mises en cache : relues dans les 5 minutes, elles coûtent 10 % du prix normal ;
- les images, PDF et fichiers ne sont payés qu'une fois, au message qui les contient ;
- `claude-haiku-5-5` coûte environ 20 fois moins cher que `claude-sonnet-5-5` ; `THINKING=off` et un `EFFORT` bas réduisent aussi les réponses ;
- chaque page de PDF coûte souvent plusieurs milliers de tokens, et chaque recherche web est payante en plus.

Le fichier `bot.log` (`LOG_FILE`) garde l'activité du bot, avec pour chaque appel les tokens utilisés et le coût estimé en dollars. Les tarifs de Sonnet, Haiku et Opus 5.5 sont connus du bot ; pour un autre modèle, renseigne `PRICE_INPUT_PER_MTOK` et `PRICE_OUTPUT_PER_MTOK`.

## Fichiers créés par le bot

À côté de l'historique (`HISTORY_FILE`), le bot crée :

| Fichier | Contenu |
| --- | --- |
| `conversation_history.json` | Mémoire de chaque salon. Si le fichier est illisible, il est mis de côté sous un nom daté, jamais écrasé |
| `waiting_messages.json` | Messages reçus pendant une absence, qui attendent leur réponse |
| `presence_sources.json` | Listes d'animes, de musiques et de jeux, mises à jour une fois par jour |
| `presence_state.json` | Statut Discord en cours, repris après un redémarrage |
| `slash_commands.sha256` | Empreinte des commandes slash, pour ne les envoyer à Discord que quand elles changent |
| `bot.log` | Logs (5 Mo par fichier, 2 anciens fichiers gardés) |

Le détail de chaque version est dans `changelog.txt`.

## Organisation du code

```
main.py                  lancement et logs
bot/config.py            lecture et validation du .env
bot/models.py            particularités de chaque modèle Claude (réflexion, tarifs)
bot/claude_client.py     appels à l'API de Claude, personnalité
bot/history.py           mémoire des conversations
bot/discord_bot.py       événements Discord, réponses, /reset
bot/emojis.py            emojis du serveur : liste pour Claude, noms remplacés à l'envoi
bot/presence.py          statut Discord : tirage au sort et changements
bot/presence_sources.py  animes, musiques et jeux (AniList, Deezer, Steam)
bot/images.py            préparation des images (taille, format)
bot/pdf_files.py         vérification des PDF joints (pages, mot de passe)
bot/text_files.py        lecture des fichiers texte joints
bot/text_utils.py        découpage des longues réponses, réflexion affichée retirée
```
