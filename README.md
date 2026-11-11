# Chatbot Discord (Claude)

Un bot Discord qui se comporte comme un membre du serveur : il discute dans un salon qui lui est réservé, répond quand on le mentionne ailleurs, et a une vraie présence sur le serveur (il regarde des animes, écoute de la musique, s'absente…). Il est propulsé par Claude, le modèle d'IA d'Anthropic, et sa personnalité se règle dans un simple fichier texte.

Le bot parle français. Sa façon de parler, son caractère et son nom sont ceux que tu lui donnes dans sa personnalité.

## Fonctionnalités

### Conversation

- **Salon dédié** : le bot répond à tous les messages du salon choisi (`CHANNEL_ID`). Il se souvient des derniers échanges, même après un redémarrage. Il attend environ 2 secondes avant de répondre (`REPLY_PAUSE`) : les messages envoyés à la suite, ou pendant qu'il écrit, reçoivent une seule réponse groupée, comme le ferait quelqu'un qui lit plusieurs messages d'un coup ; s'ils sont trop lourds ensemble (images, PDF ou fichiers texte), il y répond en plusieurs fois, chaque réponse citant son message. Si un appel échoue, chaque auteur concerné en est prévenu.
- **Mentions ailleurs** : dans les autres salons, il répond quand on le mentionne, en lisant les derniers messages pour comprendre la discussion (sans mémoire d'un appel à l'autre). Comme dans son salon, les mentions reçues dans un salon pendant qu'il écrit reçoivent une seule réponse groupée.
- **Messages cités et transférés** : avec « Répondre », il lit le message cité, même ancien, ses images et ses fichiers texte (qui partagent les 100 Ko avec ceux du message). Il lit aussi les messages transférés dans son salon.
- **Messages modifiés ou supprimés** : sa mémoire suit. Un message supprimé est oublié (et s'il l'est avant la réponse, le bot n'y répond pas), un message modifié prend son nouveau texte. Ses propres messages aussi : supprimés par lui ou par un modérateur, il sait seulement qu'ils ont existé. Si sa réponse tenait en plusieurs messages et qu'un seul est supprimé, il en oublie toute la réponse quand il ne retrouve plus le texte du message (message ancien, après un redémarrage).
- **Ses propres messages** : le bot peut modifier ou supprimer l'un de ses messages récents dans le salon où il parle (parmi les 50 derniers messages), par exemple pour cacher un spoiler sous une balise ou retirer une erreur, quand on le lui demande ou qu'il s'en rend compte. Il ne touche jamais aux messages des autres. Sa mémoire garde le nouveau texte, avec un repère qui lui rappelle qu'il l'a modifié. Au plus 3 modifications ou suppressions par réponse ; chacune demande un court appel en plus.
- **Code** : un long bloc de code (25 lignes ou 1 500 caractères au moins) part en fichier joint sous le message du bot, nommé par lui (`index.html`, `style.css`…), au lieu d'une suite de messages. Sa mémoire n'en garde que le nom : pour le faire modifier, on cite son message ou on lui renvoie le fichier.
- **Recherche web** (en option, `WEB_SEARCH=on`) : il peut chercher sur Internet et lire un lien partagé quand la question porte sur l'actualité ou des informations précises.
- **Autres bots** : il les ignore, sauf quand ils le mentionnent ou répondent à l'un de ses messages : deux bots peuvent ainsi discuter ensemble. Pour éviter une boucle sans fin (et un appel à Claude à chaque tour), il ne répond pas à plus de 5 messages de bots d'affilée dans un salon ; le compte repart de zéro au premier message d'un membre, ou après 10 minutes sans message de bot.

### Pièces jointes

Dans son salon comme dans les mentions, le bot lit jusqu'à 4 images ou un PDF, et des fichiers texte, par message :

- **Images** (JPEG, PNG, GIF, WebP, BMP, TIFF) : réduites et converties automatiquement si besoin, 40 millions de pixels au plus (120 millions pour la plupart des photos JPEG). Avec plusieurs images, il peut les comparer ou parler de « la deuxième ».
- **PDF** : texte, tableaux, schémas et pages scannées, 20 pages (`PDF_MAX_PAGES`) et 20 Mo au plus. Les PDF protégés par un mot de passe d'ouverture sont refusés ; ceux protégés seulement contre la modification ou l'impression sont lus.
- **Fichiers texte** (`.txt`, `.md`, `.csv`, `.json`, `.log`, sous-titres, configuration, HTML, code…) : autant qu'on veut dans un même message, 100 Ko au total.
- **GIF du sélecteur de Discord** : pour un GIF Klipy ou un favori gardé par Discord, le bot voit 6 moments de l'animation, du début à la fin, réunis en une seule image ; pour les autres (Giphy, anciens favoris Tenor), seulement la première image. Une image partagée par un lien Discord (par exemple une photo déjà postée ailleurs) est vue elle aussi.

Le reste (vidéos, images au-delà de la 4e, PDF accompagné d'images ou deuxième PDF, autres types de fichier) n'empêche pas le bot de répondre : il est ignoré, et le bot sait ce qu'il n'a pas vu, pour pouvoir le dire.

Seul le nom du fichier reste ensuite en mémoire : le bot se souvient qu'il l'a lu, sans en repayer le contenu à chaque message.

### Vie sur le serveur

- **Statut Discord** (`PRESENCE=on`) : sous son pseudo, le bot regarde quelques épisodes d'un anime (top 500 d'AniList), écoute quelques morceaux (tops Deezer), joue à un jeu des 500 meilleures ventes de Steam en France, affiche son humeur du moment, ou passe absent ou en « Ne pas déranger » avec une raison qu'il écrit lui-même. Pour varier, un type de statut ne revient pas avant au moins deux autres. Il dort la nuit (`PRESENCE_NIGHT`) et sait ce qu'il est en train de faire quand on lui en parle. Après un redémarrage, il reprend son statut là où il en serait s'il ne s'était pas arrêté.
- **Absences** : absent, il ne répond qu'à son retour ; en « Ne pas déranger », au bout de 5 à 15 minutes. S'il s'en va en pleine conversation, il prévient d'abord son salon. Les messages en attente ou en cours de réponse sont gardés même après un redémarrage, avec un seul nouvel essai au démarrage. Au-delà de 100 Mo de fichiers en attente, un nouveau message avec fichier est refusé, et son auteur est invité à le renvoyer plus tard.
- **Messages spontanés** (`SPONTANEOUS_PER_DAY`, désactivés par défaut) : quand il finit son anime ou sa partie de jeu et qu'aucun membre n'a écrit dans son salon depuis 3 heures, le bot en parle parfois de lui-même (une fois sur trois environ), en s'adressant éventuellement à un membre que le sujet intéresse d'après ses notes. Il peut aussi choisir de ne rien dire. Jamais la nuit, au plus `SPONTANEOUS_PER_DAY` fois par 24 heures, et jamais deux fois de suite sans qu'un membre ait écrit entre-temps (les messages des bots ne comptent pas).
- **Emojis du serveur** : le bot connaît automatiquement tous les emojis du serveur, animés compris, et peut les utiliser dans ses réponses. Un emoji ajouté ou retiré est pris en compte tout de suite.

### Mémoire des membres

Avec `MEMORY=on` :

- **Ce qui est retenu** : goûts, projets, habitudes, y compris ce que les autres membres racontent, avec leur nom comme source (« d'après Toto ») : le bot garde l'identifiant Discord de Toto, pas son pseudo, et affiche toujours son nom actuel. Chaque fichier garde au plus 25 lignes : le bot n'y garde que l'essentiel. Il ne peut pas retirer d'un coup plus de la moitié d'un fichier : effacer se fait avec `/memoire` ou `/memoire-serveur`. La personne concernée le voit dans sa fiche, même si Toto n'a pas activé la mémoire ou en a parlé dans un salon qu'elle ne voit pas. Jamais de confidences ni de jugements ; une information sensible (santé, religion, opinions, adresse, argent…) seulement si le membre la donne lui-même, et le bot ne l'évoque ensuite qu'avec lui. Il se sert de ce qu'il sait dans la conversation, mais ne récite pas la fiche d'un membre aux autres.
- **Activation** : la mémoire est désactivée pour chacun par défaut. Un membre qui le souhaite l'active lui-même avec `/memoire`, après avoir lu ce qu'elle retient et qui peut voir sa fiche. Rien n'est noté sur un membre qui ne l'a pas activée, même quand les autres parlent de lui. Le bot n'envoie aucun message pour la présenter : c'est à vous d'en parler sur le serveur.
- **`/memoire`** : chacun voit la fiche que le bot tient sur lui, dans un message visible par lui seul. Une fiche se modifie ligne par ligne : le membre peut y ajouter des lignes (bouton « Ajouter des lignes »), en corriger une (bouton « Modifier une ligne », qui ouvre son texte dans une fenêtre) et en supprimer en cochant leurs numéros, sous la liste des lignes affichées en entier. Il écrit aussi lui-même ses surnoms (bouton « Surnoms »). Il peut aussi l'effacer (après confirmation) ou désactiver la mémoire : sa fiche est alors effacée, il est remplacé par « un membre » là où il est cité comme source dans les autres fiches (pour les notes plus anciennes, écrites avec son nom plutôt que son identifiant : « d'après … », « selon … » ; un nom en lettres décorées est aussi cherché sous sa forme simple ; un nom aussi porté par un autre membre est laissé, pour ne pas effacer les sources de cet autre), et plus rien n'est noté sur lui, jusqu'à ce qu'il la réactive. Le fichier général n'est pas modifié : le bot a pour consigne de ne rien y noter sur les membres, et le membre est invité à demander à un administrateur de le vérifier avec `/memoire-serveur`. Dans les deux cas, ses derniers messages restent dans l'historique de la conversation. Les administrateurs voient la fiche de n'importe quel membre avec `/memoire membre:@…`, et peuvent en supprimer des lignes ou l'effacer ; seul le membre lui-même peut l'activer ou la désactiver. Exception : pour un membre parti du serveur (désigné par son identifiant), qui ne peut plus lancer `/memoire`, un administrateur peut désactiver la mémoire à sa place, ce qui efface tout comme s'il l'avait fait lui-même. Les membres en sont prévenus (avant d'activer la mémoire, et sous leur fiche dans `/memoire`), et chaque ouverture de la fiche d'un autre est notée dans le log (qui, quelle fiche, sans le contenu).
- **`/memoire-serveur`** : le fichier général du serveur, à voir ou effacer. Les administrateurs, seuls à y avoir accès, peuvent y ajouter, modifier ou supprimer des lignes. Un membre cité avec « @ » et son nom d'utilisateur (« d'après @alice ») est retenu comme une source notée par le bot : son nom actuel est toujours affiché.

### Modération

- **`/reset`** : efface la mémoire de conversation du salon du bot, quel que soit le salon où la commande est lancée. Réservé aux membres qui ont la permission Administrateur.
- **`/skip`** (avec `PRESENCE=on`) : abandonne le statut Discord en cours et en tire un nouveau tout de suite ; seul celui qui lance la commande voit le nouveau statut. Absent, le bot répond alors aux messages en attente s'il passe en ligne. Sans effet la nuit. Réservé aux mêmes personnes que `/reset`.
- Le bot ne peut jamais pinger `@everyone`, `@here`, un rôle ou un membre.

## Avant de commencer

Il te faut :

- **Python 3.11 ou plus récent** (ou Docker, voir plus bas) ;
- **une application Discord** avec un bot (voir l'étape suivante) ;
- **une clé API Anthropic** : elle se crée sur https://platform.claude.com, rubrique API Keys. L'API est payante à l'usage, séparément de l'abonnement Claude Pro (voir [Coûts](#coûts-et-logs)).

## Créer le bot sur Discord

1. Sur https://discord.com/developers/applications, crée une application (**New Application**) et donne-lui le nom du bot.
2. Onglet **Installation** : dans **Installation Contexts**, décoche **User Install** (garde **Guild Install**), et choisis **None** comme **Install Link**.
3. Onglet **Bot** :
   - décoche **Public Bot** : sinon n'importe qui peut inviter le bot sur son serveur (Discord le refuse tant que l'Install Link de l'étape 2 n'est pas sur **None**) ;
   - clique sur **Reset Token** et copie le token : c'est le `DISCORD_TOKEN` (à garder secret) ;
   - dans **Privileged Gateway Intents**, active **Message Content Intent**. Les intents « Presence » et « Server Members » ne servent pas.
4. Onglet **OAuth2 > URL Generator** :
   - coche les scopes **bot** et **applications.commands** (ce dernier pour les commandes comme `/reset` et `/memoire`) ;
   - dans les permissions, coche **View Channels**, **Send Messages** et **Read Message History** ;
   - ouvre l'adresse générée et invite le bot sur ton serveur.
5. Dans Discord, active le mode développeur (**Paramètres utilisateur > Avancés > Mode développeur**), puis fais un clic droit sur le salon du bot > **Copier l'identifiant du salon** : c'est le `CHANNEL_ID`.

Le bot n'a besoin d'aucune autre permission : il n'utilise que les emojis du serveur.

Le bot ne sert qu'un serveur, celui du salon `CHANNEL_ID` : ailleurs (autre serveur, messages privés), il ignore les messages et refuse les commandes, puisque sa mémoire est commune. S'il a été ajouté à un autre serveur, le log le signale au démarrage.

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
   Il annonce son arrivée dans son salon avec sa version. En cas d'erreur de configuration (variable manquante, token invalide, clé Anthropic invalide, modèle inexistant, intent non activé, `CHANNEL_ID` qui ne désigne pas un salon textuel où le bot peut écrire, permission manquante), il s'arrête en expliquant quoi corriger. Toutes les erreurs du `.env` sont signalées d'un coup, et le message est aussi écrit dans `bot.log`. Sans connexion Internet, ou si Discord est en panne, il s'arrête aussi avec un message clair. Seule exception : si l'un de ses serveurs est indisponible chez Discord au démarrage (panne), il patiente et s'annonce dès le retour du serveur.

## Configuration

Tous les réglages se trouvent dans le `.env`, et chacun est expliqué dans `.env.exemple`. Les principaux, avec la valeur proposée dans `.env.exemple` :

| Réglage | Rôle | Valeur proposée |
| --- | --- | --- |
| `CLAUDE_MODEL` | Modèle utilisé : `claude-sonnet-5-5` (recommandé), `claude-haiku-5-5` (bien moins cher) ou `claude-opus-5-5` (plus capable, plus cher) | `claude-sonnet-5-5` |
| `THINKING` | Réflexion avant de répondre : `on` (plus réfléchi, plus lent et plus cher) ou `off` (vide = `off`) | `on` |
| `EFFORT` | Niveau d'effort : `low`, `medium`, `high`, `xhigh` ou `max` (`xhigh` et `max` nécessitent `THINKING=on`, sauf sur `claude-opus-5-5`) ; vide = valeur du modèle | `medium` |
| `MAX_TOKENS` | Longueur maximale d'une réponse (réflexion et notes de mémoire comprises), en tokens ; ne coûte que si elle est atteinte. Avec la mémoire, de quoi réécrire un fichier plein après la réflexion (voir `.env.exemple`) : le bot refuse de démarrer sous le minimum, et prévient dans le message « En ligne » si la réflexion risque de ne pas tenir | `6000` |
| `MAX_HISTORY_LENGTH` | Messages gardés en mémoire dans le salon dédié (jusqu'à 3 fois plus pendant une conversation suivie, avant d'y revenir d'un coup ou après 5 minutes de silence, ce qui garde la mémoire en cache) | `10` |
| `CONTEXT_MESSAGE_LIMIT` | Messages lus autour d'une mention (100 au plus) | `20` |
| `REPLY_PAUSE` | Secondes d'attente avant de répondre dans le salon dédié, de 0 à 10, pour regrouper les messages envoyés à la suite | `2` |
| `API_TIMEOUT` | Attente maximale d'une réponse, en secondes, relances comprises (une requête trop lente n'est pas relancée) ; le bot prévient au démarrage s'il est trop court pour `MAX_TOKENS` | `90` |
| `WEB_SEARCH` | Recherche web et lecture de liens (`on` ou `off`) ; prévois alors 40 secondes de plus pour `API_TIMEOUT` (`120` avec `MAX_TOKENS=6000`) | `off` |
| `PRESENCE` | Statut Discord et absences (`on` ou `off`) ; les `PRESENCE_*` règlent les chances de chaque statut et la nuit | `on` |
| `SPONTANEOUS_PER_DAY` | Messages spontanés au plus par 24 heures, de 0 (jamais) à 10 ; nécessite `PRESENCE=on` | `0` |
| `MEMORY` | Mémoire des membres (`on` ou `off`) ; `MEMORY_MEMBER_LIMIT` et `MEMORY_SERVER_LIMIT` règlent la taille des fiches et du fichier général (3000 et 5000 caractères) | `off` |
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

L'image ne contient ni `.env`, ni personnalité `*.local.txt`, ni historique, ni mémoire des membres. Les réglages se passent en variables d'environnement, et les fichiers à garder entre deux redémarrages doivent être sur un volume. Le dossier `memories/` se crée à côté de `HISTORY_FILE`, donc sur le même volume. Par exemple :

```
docker run -d --name chatbot \
  --env-file .env \
  -e HISTORY_FILE=/data/conversation_history.json \
  -e LOG_FILE=/data/bot.log \
  -v chatbot-data:/data \
  <compte>/chatbot-maisonnette:latest
```

`--env-file` ne lit pas le `.env` comme le bot : il ne retire pas les commentaires en fin de ligne. Écris donc les valeurs sans guillemets et mets les commentaires sur leur propre ligne. Hors Docker, le fichier `.env` est prioritaire : une variable passée en ligne de commande (`MEMORY=on python main.py`) est ignorée si le `.env` la définit aussi.

Pour une personnalité personnelle, monte le fichier dans le conteneur et indique son chemin dans `PERSONALITY_FILE`.

## Coûts et logs

Chaque message envoyé à Claude est facturé selon le nombre de tokens lus (personnalité, mémoire, message) et écrits (la réponse). Pour limiter la facture :

- la personnalité et la mémoire sont mises en cache : relues dans les 5 minutes, elles coûtent 10 % du prix normal (5 % sur Opus 5.5) ;
- les images, PDF et fichiers ne sont payés qu'une fois, au message qui les contient ;
- `claude-haiku-5-5` coûte environ 20 fois moins cher que `claude-sonnet-5-5` ; `THINKING=off` et un `EFFORT` bas réduisent aussi les réponses ;
- chaque page de PDF coûte souvent plusieurs milliers de tokens, et chaque recherche web est payante en plus ;
- avec `MEMORY=on`, les fiches jointes (5 au plus par message) sont relues à chaque message, hors cache : environ 850 tokens par fiche pleine de 3000 caractères. Le prompt ne liste que les membres qui ont une fiche, mais chaque nouvelle fiche, changement de pseudo ou note dans le fichier général fait réécrire la conversation en cache au message suivant. Chaque note demande en plus un court appel, la conversation étant alors relue en cache.

Le fichier `bot.log` (`LOG_FILE`) garde l'activité du bot, avec pour chaque appel les tokens utilisés et le coût estimé en dollars. `LOG_LEVEL=DEBUG` le rend plus détaillé pour enquêter sur un problème. En conteneur, `LOG_FILE=off` n'écrit que sur la console, déjà collectée par Docker ou OpenShift. Au démarrage, le journal rappelle les réglages effectifs : modèle, `MAX_TOKENS`, `API_TIMEOUT`, et chemins complets des fichiers. Les tarifs de Sonnet, Haiku et Opus 5.5 sont connus du bot ; pour un autre modèle, renseigne `PRICE_INPUT_PER_MTOK` et `PRICE_OUTPUT_PER_MTOK`.

## Fichiers créés par le bot

À côté de l'historique (`HISTORY_FILE`), le bot crée :

| Fichier | Contenu |
| --- | --- |
| `conversation_history.json` | Mémoire de chaque salon. Si le fichier est illisible, il est mis de côté sous un nom daté, jamais écrasé |
| `waiting_messages.json` | Messages qui attendent leur réponse, repris après un redémarrage |
| `presence_sources.json` | Listes d'animes, de musiques et de jeux, mises à jour une fois par jour |
| `presence_state.json` | Statut Discord en cours, repris après un redémarrage |
| `spontaneous_messages.json` | Avec `SPONTANEOUS_PER_DAY` : derniers messages spontanés, pour respecter la limite par jour après un redémarrage |
| `slash_commands.sha256` | Empreinte des commandes slash, pour ne les envoyer à Discord que quand elles changent |
| `memories/registre.json` | Avec `MEMORY=on` : membres vus par le bot, leurs noms et anciens pseudos, et leurs réglages de la mémoire. Si le fichier est illisible, la mémoire des membres est coupée (comme avec `MEMORY=off`) et le fichier n'est pas modifié : corrige-le d'après le log, puis redémarre le bot. Le supprimer repart de zéro : la mémoire est désactivée pour tous, chacun la réactive avec `/memoire`. À la mise à jour vers la mémoire à activer soi-même, elle reste activée pour les membres qui avaient déjà une fiche et ne l'avaient pas désactivée |
| `memories/membres/<id>.md`, `memories/serveur.md` | Avec `MEMORY=on` : fiche de chaque membre et fichier général du serveur, tenus par Claude, en UTF-8. Un fichier illisible (enregistré dans un autre encodage, par exemple) est signalé dans le log : le bot continue de répondre, mais n'y écrit plus rien tant qu'il n'est pas corrigé. Après une modification directe sur le disque, redémarre le bot pour que la liste des membres (fiches existantes et surnoms) en tienne compte ; `/memoire` n'a pas besoin de redémarrage |
| `memories/modifications.json` | Avec `MEMORY=on` : date de la dernière modification faite avec `/memoire` ou `/memoire-serveur` sur chaque fichier, montrée à Claude pour qu'il ne renote pas d'après la conversation ce qui a été retiré. S'il est illisible ou supprimé, ces dates sont oubliées, sans rien bloquer |
| `bot.log` | Logs (5 Mo par fichier, 2 anciens fichiers gardés) |

Le détail de chaque version est dans [`CHANGELOG.md`](CHANGELOG.md).

## Organisation du code

```
main.py                  lancement et logs
bot/config.py            lecture et validation du .env
bot/models.py            particularités de chaque modèle Claude (réflexion, tarifs)
bot/claude_client.py     appels à l'API de Claude, personnalité
bot/prompts.py           textes du prompt de Claude (personnalité par défaut, notes techniques)
bot/history.py           mémoire des conversations
bot/members.py           registre des membres (noms, anciens pseudos, réglages de la mémoire)
bot/memory.py            fiches des membres et fichier général : outil « memoire » de Claude
bot/memory_views.py      /memoire et /memoire-serveur : affichage, pages, choix des lignes à modifier ou supprimer, fenêtres, confirmations
bot/discord_bot.py       événements Discord, réponses, messages cités ou transférés, réactions
bot/own_messages.py      messages du bot : outil « mes_messages » de Claude, pour les modifier ou les supprimer
bot/commands.py          commandes slash (/reset, /skip, /memoire, /memoire-serveur)
bot/attachments.py       pièces jointes des messages : vérification et téléchargement
bot/emojis.py            emojis du serveur : liste pour Claude, noms remplacés à l'envoi
bot/presence.py          statut Discord : tirage au sort et changements
bot/presence_sources.py  animes, musiques et jeux (AniList, Deezer, Steam)
bot/spontaneous.py       messages spontanés : quand parler de soi-même, et de quoi
bot/images.py            préparation des images (taille, format)
bot/gifs.py              GIF du sélecteur de Discord : téléchargement et grille de l'animation
bot/pdf_files.py         vérification des PDF joints (pages, mot de passe)
bot/text_files.py        lecture des fichiers texte joints
bot/media.py             pièce jointe prête pour Claude (image ou PDF)
bot/errors.py            base des erreurs dont le message s'affiche sur Discord
bot/text_utils.py        découpage des longues réponses, réflexion affichée retirée
bot/code_files.py        longs blocs de code des réponses : fichiers joints, résumés en mémoire
bot/files.py             écriture des fichiers sur le disque, sans fichier à moitié écrit
tests/                   tests automatisés : python -m unittest
```
