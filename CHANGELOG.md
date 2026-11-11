# Changelog — Chatbot Discord

Les versions les plus récentes sont en haut. Chaque entrée commence par un résumé en gras, suivi du détail.

**Versions :** [5.18](#version-518) · [5.17](#version-517) · [5.16](#version-516) · [5.15](#version-515) · [5.14](#version-514) · [5.13](#version-513) · [5.12](#version-512) · [5.11](#version-511) · [5.10](#version-510) · [5.9](#version-59) · [5.8](#version-58) · [5.7](#version-57) · [5.6](#version-56) · [5.5](#version-55) · [5.4.0](#version-540) · [5.3.0](#version-530) · [5.2.0](#version-520) · [5.1.0](#version-510-1) · [5.0.0](#version-500) · [Avant la 5.0](#avant-la-version-50)

---

## Version 5.18

### Corrections

- **Dans un message transféré, Rika reconnaît les membres, rôles et salons mentionnés** comme dans un message normal.
- **Rika sait ce qu'il est arrivé à ses propres messages** : elle se souvient de les avoir modifiés, au lieu de croire n'y avoir jamais touché, et oublie ceux qui ont été supprimés, par elle-même ou par un modérateur.
- **`/reset` vide la mémoire de Rika depuis n'importe quel salon**, au lieu de ne rien effacer quand il était lancé ailleurs que dans son salon.
- **Dans son salon, Rika répond aux messages accompagnés d'une vidéo, de plus de 4 images ou d'un PDF en trop**, comme dans les mentions, au lieu de les refuser : elle lit ce qu'elle peut et sait ce qu'elle n'a pas vu.
- **Quand un message dépasse ce que Rika peut lire en une fois, elle le dit clairement** et propose de raccourcir le message ou ses fichiers, au lieu de conseiller un `/reset` qui n'y changeait rien.

---

## Version 5.17

*Publiée le 10 novembre 2026*

> **En bref :** des jeux connus dans le statut, du code envoyé en fichier joint, plusieurs fichiers texte lus d'un coup, et une Rika qui peut retoucher ses propres messages.

### Changements

- **Rika joue à des jeux connus**, choisis parmi les 500 meilleures ventes de Steam en France au lieu de toute la boutique. `STEAM_API_KEY` ne sert plus et peut être retirée.
- **Un long morceau de code écrit par Rika arrive en fichier joint**, sous son message, au lieu d'une suite de messages. Pour le faire modifier ensuite, il suffit de citer son message (« Répondre ») ou de lui renvoyer le fichier.
- **Rika lit plusieurs fichiers texte dans un même message**, jusqu'à 100 Ko au total, y compris ceux d'un message cité, et bien plus de formats de code et de configuration (`.scss`, `.svg`, `.vue`, `.conf`, `.env`, `.scala`, `.dart`…).
- **Rika peut modifier ou supprimer ses propres messages récents**, par exemple pour cacher un spoiler sous une balise ou retirer une erreur, quand on le lui demande ou qu'elle s'en rend compte.

### Corrections

- **Une réponse où Rika atteint sa limite de notes en mémoire coûte moins cher.**
- **Une longue réponse contenant un petit bout de code ne s'affiche plus en partie comme du code.**
- **Rika ne regarde plus un épisode d'anime qui n'est pas encore sorti.**
- **Le statut de Rika ne reste plus figé quand AniList, Deezer ou Steam répond de façon inattendue.**

---

## Version 5.16

*Publiée le 8 novembre 2026*

> **En bref :** une Rika moins chère dans les longues conversations, et des GIF favoris enfin visibles.

### Changements

- **Rika coûte moins cher dans une conversation animée, et se souvient de plus de messages tant qu'elle ne s'arrête pas.** Sa mémoire monte jusqu'à 3 fois `MAX_HISTORY_LENGTH`, et revient à cette taille après 5 minutes de silence.

### Corrections

- **Rika voit aussi les GIF envoyés depuis les favoris et les images partagées par un lien Discord**, au lieu de dire qu'elle n'en voit que le lien.

---

## Version 5.15

*Publiée le 7 novembre 2026*

> **En bref :** des fiches de mémoire qu'on corrige soi-même, des GIF enfin visibles, et une Rika qui peut lancer la conversation d'elle-même.

### Changements

- **Chacun peut modifier une ligne de sa fiche dans `/memoire`**, et les administrateurs celles du fichier général dans `/memoire-serveur`.

- **Rika répond avec des mots à un sticker ou un emoji envoyé seul**, au lieu de renvoyer un sticker ou un emoji du serveur au hasard.

- **Rika peut lancer la conversation d'elle-même** quand elle finit son anime ou sa partie de jeu et que le salon est calme depuis 3 heures. Désactivé par défaut : `SPONTANEOUS_PER_DAY` règle le nombre maximum de messages par jour.

- **Rika voit les GIF choisis dans le sélecteur de Discord**, et suit l'animation du début à la fin pour ceux de Klipy. Pour les autres (Giphy, anciens favoris Tenor), elle n'en voit que la première image.

- **Le statut personnalisé de Rika dit son humeur du moment**, plus ce qu'elle est en train de faire.

### Corrections

- **Rika ne répond plus à un message supprimé avant sa réponse**, et ne le garde pas en mémoire.

- **Dans `/memoire` et `/memoire-serveur`, les lignes à supprimer se lisent en entier et se cochent par leur numéro**, et un long fichier général s'affiche en plusieurs pages au lieu d'être coupé.

- **Avec Opus 5.5, le coût estimé dans le journal n'est plus surévalué.**

---

## Version 5.14

*Publiée le 7 novembre 2026*

> **En bref :** une mémoire plus sûre, des messages qui ne se perdent plus aux redémarrages, et une seule réponse aux messages envoyés à la suite.

### Changements

- **Les fiches de mémoire se modifient ligne par ligne, et chacun peut compléter la sienne.** Dans `/memoire`, le membre ajoute ou supprime des lignes et écrit ses surnoms ; l'administrateur fait de même dans `/memoire-serveur`.

- **Une source (« d'après Alice ») suit le membre, même s'il change de pseudo.**

- **Désactiver la mémoire ne touche plus au fichier général.** `/memoire` invite le membre à le faire vérifier par un administrateur.

- **Une seule réponse aux messages envoyés à la suite dans le salon dédié.** Rika attend environ 2 secondes avant de répondre (`REPLY_PAUSE`).

- **Les anciennes fiches des membres sans mémoire sont effacées au premier démarrage.** Rika ne s'en servait plus, et personne ne pouvait les lire.

- **Python 3.11 minimum, et versions des dépendances figées.** Hors Docker, relance `pip install -r requirements.txt`.

- **Les fichiers d'avant la 5.8 et la 5.11 ne sont plus convertis.** Depuis une version aussi ancienne, passe d'abord par la 5.13.

- **Des consignes plus courtes pour Claude : environ un quart de tokens en moins par message.**

### Corrections

- **Rika ne perd plus de messages lors d'un redémarrage ou d'une panne de Discord.** Ils sont repris au démarrage suivant, et chaque auteur est prévenu si sa réponse échoue.

- **Rika ne peut plus vider une fiche sur simple demande.** Elle ne retire jamais plus de la moitié d'un fichier d'un coup ; effacer se fait avec `/memoire` ou `/memoire-serveur`.

- **Désactiver la mémoire n'efface plus les sources d'un autre membre qui avait pris le même pseudo.**

- **Les boutons de `/memoire` n'affichent plus « Échec de l'interaction » alors que l'action a réussi.**

- **Plus de « Je n'ai rien trouvé à répondre à ça » quand Rika prend beaucoup de notes d'un coup.**

- **Les gros messages ne font plus exploser le coût ni tomber le bot.** `WEB_MAX_USES` limite vraiment les recherches web, les messages groupés très longs partent en plusieurs réponses, et au-delà de 100 Mo de fichiers en attente, l'auteur est invité à renvoyer le sien plus tard.

- **Rika ne joue plus à un jeu Steam pas encore sorti.**

- **Une « version » invalide dans `registre.json` n'empêche plus le démarrage.** La mémoire des membres est alors coupée, et le log dit quoi corriger.

- **`/memoire` dit aux membres qu'un administrateur peut aussi effacer leur fiche.**

---

## Version 5.13

*Publiée le 4 novembre 2026*

> **En bref :** des pièces jointes plus solides : grandes images, PDF protégés, fichiers texte et coupures réseau.

### Changements

- **Nouvelles dépendances : `cryptography` et Pillow 11 minimum.** Hors Docker, relance `pip install -r requirements.txt`.

- **Le journal dit pourquoi un bot qui mentionne Rika est ignoré.**

### Corrections

- **Une image aux dimensions énormes ne fait plus tomber le bot.** Au-delà de 40 millions de pixels, elle est refusée.

- **Une coupure réseau pendant un téléchargement ne fait plus ignorer le message.** Rika demande de réessayer.

- **Plusieurs gros fichiers envoyés pendant que Rika écrit ne font plus échouer sa réponse.** Elle répond en plusieurs fois.

- **Les PDF protégés seulement contre la modification sont lus.**

- **Les grandes images en gris 16 bits ne deviennent plus toutes blanches.**

- **Un octet abîmé ne déforme plus tous les accents d'un fichier texte.**

---

## Version 5.12

*Publiée le 3 novembre 2026*

> **En bref :** plusieurs images par message, des réponses groupées aux mentions, et un bot qui vérifie ses réglages au démarrage au lieu de tomber en panne plus tard.

### Changements

- **Rika lit jusqu'à 4 images par message**, y compris dans un message cité ou transféré.

- **Journal réglable.** `LOG_LEVEL` choisit le niveau de détail, et `LOG_FILE=off` n'écrit que sur la console, pratique en conteneur.

- **Le message « En ligne » affiche la version, le modèle Claude et l'heure du démarrage.**

### Corrections

- **Le bot vérifie ses réglages au démarrage, et s'arrête avec un message clair si l'un est faux.** Cela vaut pour les valeurs trop grandes (plafonds dans `.env.exemple`), les dossiers absents, `CHANNEL_ID`, la clé Anthropic, le modèle et `PRESENCE_WEIGHTS`. Si le bot refuse de démarrer après la mise à jour, le message dit quoi corriger.

- **Une réponse trop lente n'est plus relancée trois fois.** `.env.exemple` propose maintenant `MAX_TOKENS=6000` et `API_TIMEOUT=90`, que le bot vérifie au démarrage.

- **Deux bots ne tournent plus en boucle.** Rika s'arrête après 5 messages de bots d'affilée.

- **Une rafale de mentions reçoit une seule réponse groupée**, comme dans le salon dédié.

- **Un `.env` entre guillemets marche aussi sous Docker** (`docker run --env-file`).

- **Une personnalité illisible ne rend plus Rika muette.** Elle garde la dernière valide, et le journal dit quoi corriger.

- **Une longue réponse avec du code n'est plus coupée et ne fige plus le bot.**

- **Une date sans fuseau dans `presence_state.json` n'empêche plus le démarrage.**

- **« Modifier » ne signale plus de faux conflit dans `/memoire`.**

- **« Réflexion : off » n'est plus affiché à tort sur Opus 5.5.**

---

## Version 5.11

*Publiée le 1er novembre 2026*

> **En bref :** la mémoire des membres devient un choix de chacun (`/memoire`), elle est beaucoup mieux protégée contre les abus, et elle respecte davantage la vie privée.

### Changements

- **La mémoire des membres s'active soi-même avec `/memoire`.** Elle est désactivée par défaut, et le message de bienvenue en privé disparaît. Au premier démarrage, elle reste activée pour ceux qui avaient déjà une fiche.

- **`MAX_TOKENS` passe à 4000 par défaut (1500 avant)**, pour que Rika puisse réécrire le fichier général quand il est plein. Si tu l'avais réglé plus bas, le bot le signale au démarrage.

- **Premiers tests automatisés** : `python -m unittest` à la racine du projet.

### Corrections — mémoire et vie privée

- **Une info sur un membre ne va que dans sa fiche.**

- **Ce qu'on efface de sa fiche n'y revient plus**, même après avoir désactivé puis réactivé la mémoire.

- **Rika ne note plus les confidences, ses jugements, ni une information sensible qu'un membre n'a pas donnée lui-même**, et elle ne récite plus la fiche d'un membre aux autres.

- **Les membres savent qui peut lire leur fiche, et que ce qu'ils racontent sur les autres est noté avec leur nom.** Chaque ouverture de la fiche d'un autre est notée dans le log.

- **Un administrateur ne peut plus écrire la fiche d'un membre qui n'a pas activé la mémoire.**

- **Les surnoms sont reconnus où qu'ils soient dans la fiche.**

### Corrections — sécurité

- **Le bot ne sert plus que son propre serveur.** Le README explique comment décocher *Public Bot* et *User Install* dans le portail Discord.

- **Plus d'usurpation par le pseudo.** Rika distingue deux membres au même pseudo, et un surnom ne peut plus reprendre le nom d'un autre.

- **Les notes restent des notes.** Rika ne note plus de règle pour tout le monde, même présentée comme venant d'un administrateur.

- **Un membre ne peut plus vider le fichier général en désactivant sa mémoire.**

- **Le token Discord et les clés n'apparaissent plus quand la configuration est affichée.**

### Corrections — `/memoire` et fiabilité

- **`/memoire` ne promet plus d'effacer plus qu'il ne le fait.**

- **La fenêtre « Modifier » n'écrase plus les changements récents**, respecte la limite du fichier général et s'ouvre même sur un fichier trop long.

- **Des messages d'erreur compréhensibles au lieu d'« Échec de l'interaction »**, et des boutons qui se grisent au bout de 10 minutes.

- **La fiche d'un membre parti reste accessible** avec `/memoire membre:` et son identifiant.

- **La mémoire coûte moins cher et ne ralentit plus le bot sur un grand serveur.**

- **Une fiche ou un registre abîmé ne bloque plus le bot et ne réinitialise plus la mémoire.** Un `memories/registre.json` abîmé coupe la mémoire jusqu'à sa réparation, et le log dit quoi corriger.

- **Une coupure de courant ne laisse plus de fiche vide.**

### Corrections — démarrage et réponses

- **Un `CLAUDE_MODEL` ou une `TIMEZONE` mal écrits sont signalés clairement au démarrage.**

- **Plus de « […] » envoyé seul quand une réponse est coupée.**

---

## Version 5.10

*Publiée le 31 octobre 2026*

> **En bref :** Rika se souvient des membres et du serveur d'une conversation à l'autre.

### Nouveautés

- **Mémoire des membres** (`MEMORY=on`). Rika tient une fiche sur chaque membre et un fichier général du serveur (blagues récurrentes, événements, habitudes). Elle décide seule de ce qui mérite d'être retenu, quelle que soit la personne qui le lui apprend, et note la source quand l'info vient d'un autre membre (« d'après Toto »). Elle écrit en style télégraphique, une ligne par fait, sans doublon, et met à jour une ligne plutôt que d'en ajouter une. Elle ne signale pas ce qu'elle note : on le découvre plus tard, quand elle s'en souvient.

- **Ce que Rika voit.** Le bot joint lui-même à chaque message les fiches des membres concernés (auteurs, membres mentionnés, auteur du message cité), sans appel supplémentaire. Son prompt contient aussi le fichier général et la liste des membres, avec tous leurs noms (pseudo, nom d'utilisateur, surnoms notés en tête de fiche, anciens pseudos) et le chemin de leur fiche : elle peut ainsi noter une info sur quelqu'un cité par son seul nom ou surnom, et lire sa fiche si besoin. Les fiches jointes ne sont pas enregistrées dans l'historique, et le cache s'arrête juste avant elles : l'historique reste relu en cache.

- **Comment elle écrit.** Avec un outil « memoire » exécuté par le bot (lire, ajouter, remplacer, réécrire), limité à ces seuls fichiers. Une fiche fait au plus 3000 caractères (`MEMORY_MEMBER_LIMIT`) et le fichier général 5000 (`MEMORY_SERVER_LIMIT`) : une note de trop est refusée, et Rika réécrit alors le fichier en plus court, en fusionnant et en retirant le moins important. Au plus 5 lectures ou écritures par réponse (`MEMORY_MAX_OPERATIONS`). Outil maison plutôt que celui d'Anthropic, qui lui ferait relire tout le dossier avant chaque réponse, soit un appel de plus à chaque message.

- **`/memoire`.** Chaque membre voit sa fiche, visible par lui seul, et peut la modifier dans une fenêtre déjà remplie, tout effacer ou désactiver la mémoire (avec confirmation : sa fiche est effacée, plus rien n'est retenu sur lui ni joint à ses messages). Il peut la réactiver au même endroit. Les administrateurs (permission Administrateur) voient, modifient et effacent la fiche de n'importe qui avec `/memoire membre:@…` ; seul le membre lui-même peut désactiver sa mémoire. `/memoire-serveur`, réservé aux administrateurs, fait de même pour le fichier général.

- **Présentation.** Au premier message d'un membre, le bot lui envoie une seule fois un message privé qui explique ce qu'il peut retenir et comment le voir, le modifier ou le désactiver avec `/memoire`. Si ses messages privés sont fermés, le message est posté dans le salon, avec son nom en mention mais sans notification, et s'efface au bout d'une minute. Les membres déjà actifs sont prévenus à leur premier message après le passage à la 5.10 ; un membre qui a déjà utilisé `/memoire` ne l'est pas.

- **Registre des membres.** Le bot garde, pour chaque membre vu, son nom d'utilisateur, son nom d'affichage, son pseudo sur le serveur et ses 5 derniers anciens pseudos, mis à jour à chaque message, ainsi que ses réglages de la mémoire. Tout est dans le dossier `memories/`, à côté de l'historique (`MEMORY_DIR` pour un autre dossier) : `registre.json`, `serveur.md` et `membres/<id>.md`. Avec Docker, il est donc sur le même volume que l'historique.

- **Mise en service.** `MEMORY=on` dans le `.env` (ou la ConfigMap), sans nouvelle permission Discord. Avec `MEMORY=off` (par défaut), rien ne change, et `/memoire` et `/memoire-serveur` indiquent que la mémoire est coupée. `MEMORY_MEMBER_LIMIT` et `MEMORY_SERVER_LIMIT` ne peuvent pas dépasser 20 000 (taille maximale de la fenêtre de modification).

### Divers

- **`/reset` et `/skip` réservés aux administrateurs** (permission Administrateur). Avant, les modérateurs avec « Gérer les messages » y avaient aussi accès. Les commandes sont renvoyées à Discord au premier démarrage de la 5.10.

---

## Version 5.9

*Publiée le 31 octobre 2026*

> **En bref :** tous les emojis du serveur, un statut Discord qui survit aux redémarrages, et la commande `/skip`.

### Nouveautés

- **Emojis du serveur.** Rika reçoit automatiquement la liste des emojis du serveur et peut tous les utiliser, sans qu'il faille les recopier dans la personnalité ; un emoji ajouté ou retiré est pris en compte tout de suite. Elle les écrit par leur nom (`:hap:`), et le bot les remplace par l'emoji avant l'envoi : plus d'identifiant inventé, et environ 3 fois moins de tokens par emoji. Les emojis animés sont compris (un bot n'a pas besoin de Nitro) ; seuls ceux réservés à des rôles qu'elle n'a pas sont exclus, y compris quand elle répond à un emoji seul. Dans sa mémoire et dans les messages qu'elle lit, les emojis apparaissent aussi par leur nom.

  La personnalité dit seulement quand en mettre, par exemple : « Tu termines chaque message par un seul emoji du serveur, directement après le dernier mot. » Une ancienne liste de codes (`<:hap:123…>`) fonctionne encore, mais n'est plus utile.

- **Statut Discord gardé entre deux redémarrages.** Rika reprend son statut là où elle en serait si elle ne s'était pas arrêtée, comme s'il avait continué pendant l'arrêt. Après un redémarrage rapide, elle regarde toujours le même épisode ou joue au même jeu ; après un arrêt plus long, l'anime a avancé d'autant d'épisodes et la playlist d'autant de morceaux. Absente ou en « Ne pas déranger », elle le reste : les messages en attente gardent leur réponse différée. Un statut terminé pendant l'arrêt, coupé par la nuit ou désactivé depuis dans `PRESENCE_WEIGHTS` laisse place à un nouveau, et ses derniers statuts personnalisés ne sont pas répétés. Le statut est gardé dans `presence_state.json`, à côté de l'historique.

- **`/skip`** abandonne le statut Discord en cours (l'anime ou la playlist entière) et en tire un nouveau tout de suite, selon les mêmes règles. Réservé aux modérateurs et administrateurs, comme `/reset` ; seul celui qui lance la commande voit le nouveau statut et sa durée. Si Rika était absente et passe en ligne, elle répond aussitôt aux messages en attente ; si elle s'éloigne en pleine conversation, elle prévient son salon. Sans effet la nuit, et absente avec `PRESENCE=off`. La commande est envoyée à Discord au premier démarrage de la 5.9.

- **Statuts plus variés.** Un nouveau statut ne peut plus être du même type que les deux derniers (avant, seul le dernier était exclu), et le type d'avant revient deux fois moins souvent. Les suites comme anime / musique / anime deviennent impossibles : un type vu dans les 3 derniers statuts revient dans 15 % des tirages au lieu de 37 %, et les chances de `PRESENCE_WEIGHTS` restent presque respectées. Si trop de statuts sont désactivés pour respecter la règle, elle s'assouplit. Les derniers types tirés sont gardés dans `presence_state.json`.

### Divers

- **Image Docker : Python 3.15** au lieu de 3.13. Toutes les dépendances ont leurs paquets précompilés pour cette version.

---

## Version 5.8

*Publiée le 31 octobre 2026*

> **En bref :** Rika tient compte des messages modifiés ou supprimés.

### Nouveautés

- **Messages modifiés ou supprimés.** Dans son salon, Rika oublie le contenu d'un message supprimé (texte, pièces jointes) et sa citation dans les réponses qui le citaient. Elle se souvient seulement que quelqu'un a écrit un message à ce moment-là, supprimé depuis, et garde sa propre réponse : la conversation ne perd pas son fil. Un message modifié prend son nouveau texte, avec une mention indiquant qu'il a été modifié après sa réponse.

  Un message supprimé avant qu'elle y réponde (pendant une absence par exemple) ne reçoit plus de réponse, y compris une mention hors de son salon ; modifié, elle répond au nouveau texte.

- **Nouveau format de l'historique (version 3)**, plus lisible : chaque message Discord y garde son identifiant, et la date, l'auteur, la citation, le texte et les pièces jointes sont séparés. L'ancien fichier est converti automatiquement au premier lancement, sans réécrire le cache de prompt, et une copie en est gardée à côté (`conversation_history.json.v2-<date>`).

  ⚠️ Revenir à la 5.7 demande de remettre cette copie à la place du fichier : la 5.7 ne sait pas lire le nouveau format et repartirait d'une mémoire vide.

### Corrections

- **Réflexion affichée.** Quand le modèle commence sa réponse par sa réflexion (`<think>…</think>`, `<reasoning>…`, faux appel d'outil), elle est retirée avant l'envoi, et le log le signale. Seul un bloc complet tout au début est retiré : une balise plus loin dans le texte ou dans du code reste affichée. Celle déjà présente dans la mémoire est nettoyée au démarrage : en la relisant, le modèle l'imitait, et une fuite en entraînait d'autres.

---

## Version 5.7

*Publiée le 31 octobre 2026*

> **En bref :** lecture des PDF, historique 2 à 3 fois moins cher, et de nombreuses corrections.

### Nouveautés

- **Rika prévient quand elle s'en va.** Quand elle passe absente ou en « Ne pas déranger » (ou part dormir) en pleine conversation, elle prévient son salon dans un court message, en accord avec son statut. Elle ne le fait que si quelqu'un y a écrit dans les 3 dernières minutes, et après avoir fini sa réponse en cours ; elle s'en souvient ensuite.

- **PDF.** Rika lit un PDF joint au message (texte, tableaux, schémas, pages scannées), dans son salon et quand on la mentionne. Un PDF par message, sans image ; 20 pages au plus (`PDF_MAX_PAGES` dans le `.env`, 0 pour refuser les PDF) et 20 Mo. Les PDF protégés par un mot de passe sont refusés. Comme pour les images, seul le nom du fichier reste en mémoire. Nouvelle dépendance : pypdf (`pip install -r requirements.txt`).

- **Mémoire moins chère.** Quand elle est pleine, elle n'oublie plus un échange à chaque message, mais plusieurs d'un coup. Elle grandit jusqu'à `MAX_HISTORY_LENGTH` + la moitié (20 → 30 messages), puis revient à `MAX_HISTORY_LENGTH`. Entre deux coupes, le début de la conversation reste dans le cache de prompt au lieu d'être réécrit à chaque message : dans une conversation animée, l'historique coûte environ 2 à 3 fois moins cher.

- **Plus de fichiers texte.** En plus des `.txt`, Rika lit les `.md`, `.csv`, `.json`, `.log`, sous-titres (`.srt`, `.vtt`), fichiers de configuration, pages HTML et le code (`.py`, `.js`, `.java`, `.cpp`, `.cs`…).

### Corrections

- **Captures d'écran nettes.** Les captures trop grandes restent en PNG au lieu d'être converties en JPEG, qui brouillait le texte (le JPEG ne sert plus que si le PNG dépasse la limite de l'API). Les images en palette (PNG optimisés, GIF) sont réduites proprement, et non plus au plus proche voisin.

- **Erreurs de l'API mieux expliquées.** Une requête refusée n'est plus toujours présentée comme un problème de formulation (ou d'image dès qu'une image est jointe). Rika parle de l'image seulement si c'est elle qui pose problème, propose `/reset` si la conversation est trop longue, et demande de prévenir l'administrateur pour un réglage du `.env` refusé, un crédit épuisé, un modèle introuvable ou un accès refusé ; les logs en donnent la cause.

- **Fichiers « Unicode » du Bloc-notes.** Les `.txt` enregistrés en UTF-16 sont lus, au lieu d'être refusés comme « pas du texte lisible ».

- **Stickers.** Un sticker accompagné de texte ou d'une pièce jointe n'efface plus le message (même pendant une absence) : Claude reçoit le texte et le nom du sticker. « Aucun sticker utilisable » et « Aucun emoji personnalisé » ne sont plus envoyés à chaque fois : si le serveur n'a rien d'utilisable, Rika répond avec des mots.

- **Indicateur « en train d'écrire ».** Si Discord le refuse (permission manquante, panne passagère), la réponse est quand même envoyée, au lieu d'être perdue.

- **Commandes slash.** Elles ne sont plus renvoyées à Discord à chaque démarrage (Discord limite ces envois), seulement quand elles ont changé. Leur empreinte est gardée dans `slash_commands.sha256`, à côté de l'historique (le supprimer force un nouvel envoi). Un échec de l'envoi n'empêche plus le bot de démarrer : il réessaie au démarrage suivant.

- **Historique abîmé.** Un fichier `conversation_history.json` d'une forme inattendue (liste, « channels » qui n'est pas un dictionnaire, mauvais encodage) n'empêche plus le bot de démarrer : il est mis de côté et Rika repart d'une mémoire vide. Les copies de secours sont datées (`.corrompu-AAAA-MM-JJ_HH-MM-SS`) et ne s'écrasent plus entre elles.

- **Réglages du `.env` vérifiés.** `API_TIMEOUT` doit valoir au moins 1 seconde, et les prix (`PRICE_*`) ne peuvent plus être négatifs. « nan » et « inf » sont refusés pour tous les réglages décimaux, `PRESENCE_SPEED` compris. Une valeur invalide arrête le bot au démarrage avec un message clair.

---

## Version 5.6

*Publiée le 30 octobre 2026*

> **En bref :** Rika a une vie sous son pseudo Discord — elle regarde des animes, écoute de la musique, joue, et n'est pas toujours disponible.

### Nouveautés

- **Statut Discord.** Sous son pseudo, Rika regarde un anime, écoute de la musique, joue à un jeu, affiche une pensée du moment, ou passe en « Ne pas déranger » ou absente.
  - **Chances de chaque statut :** anime 20 %, musique 20 %, pensée 10 %, jeu 20 %, en ligne sans activité 20 %, Ne pas déranger 5 %, absente 5 % (`PRESENCE_WEIGHTS` dans le `.env`). Chaque statut a sa propre durée, tirée au sort ; à la fin, un nouveau statut est tiré, jamais le même que le précédent.
  - **Animes :** un titre (romaji) au hasard dans le top 500 d'AniList. Rika regarde 1 à 5 épisodes à la suite, à partir d'un épisode précis, à la durée réelle d'un épisode (un film compte pour un épisode).
  - **Musiques :** 6 à 20 titres au hasard dans le Top Worldwide, le Top France et le Top Japan de Deezer, chacun à sa durée réelle.
  - **Jeux :** un jeu au hasard dans toute la boutique Steam (clé `STEAM_API_KEY` ; sans elle, pas de statut « joue à »), pendant 45 min à 2 h. Les jeux pour adultes sont écartés, et Rika connaît la présentation du jeu sur la boutique.
  - **En ligne sans activité :** 15 à 45 min.
  - **Pensée du moment** (30 à 90 min), **raison d'une absence** (10 à 60 min) ou **d'un Ne pas déranger** (30 à 90 min) : écrites par Claude avec la personnalité du bot, selon l'heure et la durée, sans reprendre les dernières.
  - **La nuit**, de 2 h à 6 h, Rika est automatiquement absente (`PRESENCE_NIGHT`), quoi qu'elle soit en train de faire.
  - Quand on lui parle, Rika sait ce qu'elle est en train de faire, et pour combien de temps encore.
  - `PRESENCE_SPEED=10` rend tous les statuts 10 fois plus courts, pour tester.
  - Les listes sont téléchargées au démarrage puis une fois par jour, et gardées dans `presence_sources.json` (à côté de l'historique) au cas où un site ne répondrait plus.

- **Réponses différées.** Absente (et la nuit), Rika ne répond plus avant son retour ; en « Ne pas déranger », elle répond au bout de 5 à 15 minutes, ou dès qu'elle change de statut.
  - Dans son salon, tous les messages arrivés entre-temps reçoivent une seule réponse groupée. Ailleurs, elle répond à chaque mention avec « Répondre », pour qu'on sache à quel message elle répond.
  - Elle sait qu'elle répond en retard, et pourquoi (absente, Ne pas déranger, la nuit, avec la raison affichée).
  - Elle ne réagit pas aux stickers ni aux emojis seuls pendant ce temps.
  - Les messages en attente sont gardés dans `waiting_messages.json` (à côté de l'historique) : après un redémarrage, Rika y répond quand même.

---

## Version 5.5

*Publiée le 30 octobre 2026*

> **En bref :** moins de réflexion qui fuit dans les réponses, et des logs plus propres.

### Corrections

- **Avertissements vocaux masqués.** Les avertissements de discord.py sur le vocal (PyNaCl et davey non installés) n'apparaissent plus à chaque démarrage : le bot n'utilise pas le vocal.

- **Réflexion affichée.** Avec Sonnet 5.5 et `EFFORT=high`, le modèle écrivait parfois sa réflexion dans sa réponse (faux appel d'outil `<invoke name="think">…`, raisonnement entre balises `<reasoning>…`), quand il choisissait de ne pas réfléchir. Ça n'arrive plus avec `EFFORT=medium` : c'est le réglage conseillé.
  - Les consignes ajoutées automatiquement à la personnalité suivent les recommandations d'Anthropic contre ces fuites : plus d'interdiction de commenter les recherches web (ce commentaire n'est de toute façon jamais affiché), et une consigne générale contre les balises internes.
  - Au démarrage, le log indique le modèle, la réflexion, l'effort et la recherche web réellement utilisés.

- **Logs moins bavards.** Chaque requête HTTP vers l'API d'Anthropic n'est plus journalisée (le SDK 2.x passe par httpx2, qui n'était pas rendu silencieux).

---

## Version 5.4.0

*Publiée le 29 octobre 2026*

> **En bref :** choix du modèle Claude, lecture des messages transférés, et réponses groupées.

### Nouveautés

- **Changement de modèle facile.** Il suffit de modifier `CLAUDE_MODEL` dans le `.env` (`claude-sonnet-5-5`, `claude-haiku-5-5` ou `claude-opus-5-5`) et de redémarrer le bot.
  - `THINKING=off` coupe la réflexion de la façon propre à chaque modèle (auparavant, tout autre modèle que Sonnet 5.5 était refusé par l'API).
  - Sur Opus 5.5, la réflexion ne peut pas être coupée : régler `EFFORT=low`.
  - Les tarifs utilisés pour le coût estimé dans les logs suivent le modèle choisi ; `PRICE_INPUT_PER_MTOK` et `PRICE_OUTPUT_PER_MTOK` ne sont plus à renseigner que pour un autre modèle (à vider dans le `.env` existant).

- **Messages transférés.** Dans le salon dédié, le bot lit le message transféré (son texte et son image).

- **Discussion entre bots.** Le bot répond aux autres bots quand ils le mentionnent ou répondent à l'un de ses messages ; le reste de ce qu'ils postent est toujours ignoré.

- **Messages regroupés.** Dans le salon dédié, les messages envoyés pendant que le bot répond ne reçoivent plus chacun leur réponse ; ils lui sont transmis ensemble (avec leurs images) et il y répond en une seule fois.

### Corrections

- **Messages système ignorés.** Les messages système (arrivée d'un membre, message épinglé, boost, fil créé…) et les messages sans rien à lire (sondages…) ne déclenchent plus de réponse payante de Claude.

- **`/reset` pendant une réponse.** La mémoire était mal vidée : l'échange en cours y était réécrit juste après. Le reset attend désormais la fin de la réponse avant de tout effacer.

- **Mentions moins chères.** Le contexte lu pour répondre à une mention n'est plus écrit dans le cache de prompt (surcoût de 25 %), puisqu'il change à chaque mention et n'y était jamais relu.

- **Image d'une mention.** L'image jointe à la mention elle-même est maintenant lue (seule celle du message cité l'était). Si les deux en ont une, c'est celle de la mention qui est envoyée à Claude.

---

## Version 5.3.0

*Publiée le 29 octobre 2026*

> **En bref :** accès à Internet en option, et un cache qui divise le coût des messages.

### Nouveautés

- **Accès à Internet** (désactivé par défaut). Avec `WEB_SEARCH=on` dans le `.env`, le bot peut faire des recherches web et lire les liens partagés dans la conversation (outils web_search et web_fetch d'Anthropic).
  - Il ne cherche que si la question porte sur l'actualité ou une info qu'il ne connaît pas, et nomme sa source dans sa réponse (sans lien).
  - `WEB_MAX_USES` limite le nombre de recherches (et de pages lues) par message.
  - Seule la réponse finale est envoyée sur Discord, sans les commentaires écrits pendant les recherches.
  - Dans la mémoire, une mention indique au bot qu'il a réellement cherché, pour qu'il ne croie pas avoir inventé sa réponse au message suivant.
  - Avec la recherche activée, prévoir `API_TIMEOUT=90`.

### Améliorations

- **Cache de prompt.** La personnalité, les outils et l'historique sont relus depuis le cache à 10 % du prix (pendant 5 minutes, prolongées à chaque message). Avec la recherche web activée, un message simple passe d'environ 0,015 $ à 0,002 $.

- **Coût estimé plus juste.** Le coût dans les logs tient compte du cache (tokens lus et écrits) et des recherches web (`PRICE_WEB_SEARCH_PER_1000`).

---

## Version 5.2.0

*Publiée le 28 octobre 2026*

> **En bref :** lecture des fichiers texte, et des réponses plus naturelles.

### Nouveautés

- **Fichiers texte.** Le bot lit un fichier `.txt` joint au message et en tient compte dans sa réponse.
  - Un fichier `.txt` par message (en plus d'une éventuelle image), 100 Ko maximum.
  - Fonctionne dans le salon dédié et en mode mention.
  - Encodage UTF-8 ou Windows (Bloc-notes) reconnu automatiquement.
  - Dans la mémoire, seul le nom du fichier est conservé, pas son contenu (comme pour les images), pour ne pas faire grimper le coût des messages suivants.

- **Personnalité perso.** Les fichiers `*.local.txt` sont exclus de GitHub (`.gitignore`). On peut y mettre une personnalité contenant des infos privées (ex. `PERSONALITY_FILE=personality_prompt.local.txt` dans le `.env`) ; `personality_prompt.txt` reste la personnalité par défaut publiée sur GitHub.

### Améliorations

- **Réponses plus naturelles.** Le bot n'utilise plus la fonction « Répondre » de Discord pour ses réponses : il écrit simplement à la suite dans le salon, comme un humain.
  - Les messages d'erreur (fichier non pris en charge, image en trop, etc.) restent en « Répondre », pour qu'on sache à qui ils s'adressent.

---

## Version 5.1.0

*Publiée le 28 octobre 2026*

> **En bref :** le bot comprend les réponses à un message, même ancien.

### Nouveautés

- **Réponses aux messages.** Quand on utilise « Répondre » sur un message, même ancien, le bot lit le message cité et en tient compte dans sa réponse.
  - Si le message cité vient du bot, il sait qu'il s'agit de sa propre réponse.
  - Le texte cité est limité à 500 caractères.
  - Si le message cité contient une image, elle est aussi envoyée à Claude, sauf si la réponse contient déjà sa propre image (celle-ci est prioritaire).
  - Fonctionne dans le salon dédié et en mode mention.

- **Images en mode mention.** Le bot peut maintenant voir une image, celle du message cité.

### Améliorations

- **Images.** Le téléchargement et la préparation des images passent par une seule fonction, partagée entre le salon dédié et les messages cités. Une image citée illisible ou trop lourde est simplement ignorée (avertissement dans `bot.log`) au lieu de bloquer la réponse.

- **`/reset` réservé aux modérateurs** (permission « Gérer les messages ») et aux administrateurs du serveur.
  - Elle est masquée pour les autres membres et n'est plus disponible en messages privés.
  - Si un administrateur l'ouvre à tous dans les paramètres du serveur, le bot refuse quand même la commande (réponse visible uniquement par l'auteur) et note la tentative dans `bot.log`.

### Divers

- discord.py 2.5 minimum est désormais requis (`requirements.txt`).
- README : ajout de la fonctionnalité « Réponses », et précision sur les droits nécessaires pour `/reset`.

---

## Version 5.0.0

*Publiée le 28 octobre 2026, en même temps que la 5.1*

> **En bref :** le bot passe de Mistral à Claude.

### Nouveautés

- **Passage de Mistral à Claude** (Anthropic) pour générer les réponses.
- **Réécriture complète** du bot.

### Migration depuis la version 4 (Mistral)

- L'ancien `conversation_history.json` est converti automatiquement au premier lancement (les images sont remplacées par un marqueur texte).
- Dans le `.env` : `MISTRAL_API_KEY` devient `ANTHROPIC_API_KEY`, et `MISTRAL_MODEL` devient `CLAUDE_MODEL`.
- L'intent Discord « Presence » n'est plus nécessaire.

---

## Avant la version 5.0

Historique des versions précédentes, avant la réécriture du bot pour Claude. Ce code n'est plus dans le dépôt. Cliquez sur une version majeure pour l'ouvrir.

<details>
<summary><strong>Versions 4.x</strong> : passage à Mistral (4.0.0 à 4.6.8)</summary>

### Version 4.6.8

- Correction d'une erreur liée aux images corrompues.

### Version 4.6.7

- Retour à la version 4.6.3.

### Version 4.6.6

- Retour à la version 4.6.4.

### Version 4.6.5

- Il n'est plus nécessaire de citer Rika pour qu'elle réagisse : écrire son pseudonyme suffit.

### Version 4.6.4

- Le prompt de personnalité est maintenant chargé après (et non avant) l'historique des messages, ce qui peut éviter que l'IA boucle.

### Version 4.6.3

- Ajout du paramètre « Température » en tant que variable d'environnement.

### Version 4.6.2

- Modification de la manière dont sont rédigés les messages utilisateurs dans l'historique.

### Version 4.6.1

- Plusieurs valeurs écrites en dur dans le code deviennent des variables d'environnement : le modèle utilisé, la taille maximum des images, le nombre de messages comptés lors d'une citation.
- Les messages du bot de plus de 2000 caractères sont scindés en plusieurs messages au lieu de renvoyer une erreur.
- Augmentation du nombre de tokens de sortie possible : 1000 → 128000.

### Version 4.6.0

- Le bot connaît désormais la date et l'heure actuelles.
- Changement du modèle (mais pas vraiment) : `mistral-medium-2508` → `mistral-medium-latest`.
- Ajout d'un log dans la console indiquant le nombre de tokens envoyés et reçus, et le coût total du message.
- Réorganisation globale du code.

### Version 4.5.1

- Correction : quand il était cité, le bot ne voyait pas les auteurs des messages précédents.
- Messages précédents récupérés : 10 → 20.

### Version 4.5.0

- Les utilisateurs peuvent désormais citer le bot dans un autre salon que le sien.

### Version 4.4.0

- Utilisation du modèle `mistral-medium-2508` à la place de `pixtral-large-2411`.

### Version 4.3.1

- Si on envoie un fichier autre qu'une image, le bot répond par un message d'erreur.
- Augmentation de l'historique à 20 messages.

### Version 4.3.0

- La commande `!reset` est remplacée par `/reset`.
- Correction : Rika répondait par un smiley aux messages ne contenant qu'un seul mot.

### Version 4.2.0

- Amélioration de la vérification des variables d'environnement.
- Nombre d'images possibles par message : 3 → 1.
- Le bot répond par un emoji personnalisé du serveur si l'utilisateur envoie un message contenant uniquement un emoji personnalisé.
- Correction : le bot pouvait répondre à un sticker dans n'importe quel salon où il se trouvait.

### Version 4.1.1

- Taille maximale des images : 2 → 5 Mo.

### Version 4.1.0

- Modification de la manière dont sont gérés les logs.

### Version 4.0.1

- Correction de l'erreur possible lors de l'envoi d'un sticker : `discord.errors.Forbidden: 403 Forbidden (error code: 50081): Cannot use this sticker`.
  - Le sticker envoyé est désormais affiché dans la console. Si le bot ne parvient pas à l'envoyer, un autre sticker est sélectionné automatiquement.
- Ajout de l'heure dans la console pour chaque message.

### Version 4.0.0

- Refonte totale du bot à partir de zéro : il passe de l'API OpenAI à l'API Mistral.
- Suppression des messages automatiques après inactivité.
- Suppression de la fonction de rappel.
- La commande `/reset_history` devient `!reset`.
- La base de données est remplacée par un fichier `.json`.
- Si un utilisateur envoie un sticker, le bot répond par un sticker du serveur.
- Autres modifications.

</details>

<details>
<summary><strong>Versions 3.x</strong> : messages spontanés après inactivité, <code>/help</code>, rappels (3.0.0)</summary>

### Version 3.0.0

- Après une période d'inactivité, le bot envoie maintenant un message de lui-même dans son salon.
  - Période d'inactivité : entre 3 et 6 heures, choisie au hasard.
  - Cette fonction n'est pas active entre minuit et 7 h.
  - Le bot réagit à l'activité en cours d'un membre du serveur ayant le rôle « Rika ».
  - Si aucun membre n'a d'activité, le bot envoie un message d'ennui.
- La date du jour et l'heure actuelle sont envoyées à OpenAI avec le reste, pour qu'il puisse éventuellement s'en servir dans son message.
- Création de la commande `/help`, qui affiche toutes les commandes disponibles.
- Création des commandes `/mes_rappels` et `/supprimer_rappel`, qui permettent aux utilisateurs de voir et supprimer leurs rappels enregistrés.
- Nombre de messages conservés dans l'historique : 150 → 50.

</details>

<details>
<summary><strong>Versions 2.x</strong> : analyse d'images avec GPT-4o, historique persistant (2.0.0 à 2.8.0)</summary>

### Version 2.8.0

- Le bot peut séparer ses messages en plusieurs parties s'ils sont trop longs, pour qu'ils ne soient plus coupés en plein milieu.
- Ajout de la commande `/rappel`, utilisable par tous : le bot vous rappelle quelque chose à la date et l'heure choisies.
  - Le bot vous cite dans son salon à la date et l'heure indiquées.
  - Les rappels sont visibles par tous les utilisateurs.
  - Pour le moment, les rappels utilisent uniquement le fuseau horaire Europe/Paris.
- La commande `!reset_history` devient `/reset_history` (réservée aux administrateurs).

### Version 2.7.0

- Ajout de l'indicateur « Chatbot est en train d'écrire… » pendant que le bot prépare sa réponse.
- Modification de la manière dont les messages des utilisateurs sont sauvegardés dans la base de données.
- Quand plusieurs messages arrivent en même temps, ils sont traités un par un et non plus simultanément.
- Le prompt de GPT-4o pour l'analyse d'images est désormais dans un fichier séparé du code.

### Version 2.6.0

- L'historique est stocké dans une base de données MariaDB au lieu d'un fichier `.json`.
- Nombre de tokens maximum par message de GPT-4o mini : 400 → 450.
- Réorganisation du code.

### Version 2.5.0

- Modification des prompts de GPT-4o et GPT-4o mini pour les images.
- Réorganisation globale du code.

### Version 2.4.3

- Modification des prompts de GPT-4o et GPT-4o mini pour l'analyse d'images.
- L'analyse d'image de GPT-4o est conservée pendant 15 messages, ou jusqu'à ce qu'une nouvelle image soit postée.
- La réponse de GPT-4o mini à une image n'apparaît plus dans la console. Le coût en tokens apparaît à la place.
- Suppression de la fonction qui empêchait certains messages d'être sauvegardés dans l'historique. Elle ne fonctionnait pas.

### Version 2.4.2

- Modification des prompts de GPT-4o pour l'analyse d'images.

### Version 2.4.1

- Suppression du résumé des messages les plus anciens quand l'historique est plein.
- Historique : 150 → 200 messages maximum.
- Les analyses d'images de GPT-4o sont supprimées après 7 nouveaux messages.

### Version 2.4.0

- Le nom du bot est désormais une variable d'environnement.
- Meilleure intégration du prompt de personnalité lors de l'analyse d'une image.
- Ajout d'un message temporaire pendant l'analyse d'une image, pour prévenir l'utilisateur qu'elle est en cours.

### Version 2.3.0

- Maximum de messages dans l'historique : 150.
  - Arrivé à 150, les 50 messages les plus anciens sont rassemblés dans un résumé de 1000 tokens maximum.
  - À chaque nouveau palier de 150, les 50 plus anciens et le résumé précédent servent à générer un nouveau résumé.
- Le prompt de personnalité n'est plus inclus dans l'historique : il est transmis au bot à part.
- Suppression du résumé des messages longs (plus de 200 tokens).

### Version 2.2.1

- Correction : les images envoyées seules n'étaient pas commentées par le bot.

### Version 2.2.0

- Changement de modèle : GPT-4o est remplacé par GPT-4o mini.
  - Le prompt de personnalité et de contexte peut contenir bien plus d'informations.
  - L'historique de conversation peut être bien plus long.
- GPT-4o est toujours utilisé pour analyser les images, car il est plus performant pour ça.
  - Quand une image est postée, GPT-4o l'analyse et transmet son analyse à GPT-4o mini, qui commente l'image.

### Version 2.1.0

- L'historique est désormais conservé entre les redémarrages du bot.
  - Il peut être réinitialisé avec la commande `!reset_history`, réservée aux administrateurs.
- Ajout d'un message au démarrage du bot, avec sa version actuelle.
- Ajout du paramètre de température, qui influe sur la « créativité » et le côté aléatoire du bot. Valeur : 1.1.
- Suppression de la détection des dessins ASCII (qui servait à les exclure de l'historique). Elle ne fonctionnait pas.
  - Elle sera réintroduite dans une mise à jour ultérieure.
- Corrections :
  - `aiohttp.client_exceptions.ClientConnectionError: Connector is closed.`
  - `AttributeError: '_MissingSentinel' object has no attribute 'create_task'`

### Version 2.0.1

- Correction : `ValueError: unknown file extension.`

### Version 2.0.0

- Le bot peut désormais détecter les images et y réagir !
- Plus de limite maximum de messages dans l'historique. Quand l'historique atteint 30 messages, les 20 plus anciens sont résumés automatiquement.
  - Nombre maximum de tokens du résumé : 150.
  - Un log dans la console indique que l'historique a atteint 30 messages et qu'un résumé a été fait, avec le coût de l'appel à l'API.
- Les messages longs (plus de 200 tokens), qu'ils viennent de l'utilisateur ou du bot, sont résumés en 50 tokens maximum avant d'être ajoutés à l'historique.
  - Les textes longs ne sont plus gardés dans un historique temporaire de 10 messages.
  - Un log indique le coût en tokens de l'appel à l'API pour résumer le message.
- Les messages courts (moins de 5 caractères) et ceux qui ne contiennent que des emojis ne sont pas ajoutés à l'historique.
- Un log dans la console signale quand un dessin ASCII ou un texte long est détecté.
- Tokens utilisés pour générer une réponse : 500 → 400, pour que les réponses dépassent moins souvent la limite de 2000 caractères de Discord.
- Diverses optimisations du code.

</details>

<details>
<summary><strong>Versions 1.x</strong> : premières versions avec GPT-4o (1.1.0 à 1.1.2)</summary>

### Version 1.1.2

- Modification de la détection des dessins ASCII, pour qu'ils soient mieux reconnus.

### Version 1.1.1

- Correction : les textes longs de plus de 1200 caractères ou 200 caractères spéciaux n'étaient pas inclus dans la mémoire courte de 10 messages.
- Limite pour entrer dans la mémoire courte abaissée à 800 caractères ou 50 caractères spéciaux.

### Version 1.1.0

#### Nouveautés

- La conversation est commune à tous les utilisateurs, au lieu d'une conversation par utilisateur.

#### Changements

- Les images et dessins ASCII ne sont plus sauvegardés dans l'historique de l'IA, ce qui réduit la consommation de tokens.
- Les textes longs de plus de 1200 caractères ou 200 caractères spéciaux ne restent en mémoire que pendant 10 messages.
- Personnalité de l'IA modifiée et ajustée.
- L'IA connaît mieux son environnement et ses capacités.
- L'ID du salon est désormais une variable d'environnement.
- Le prompt de personnalité de l'IA est désormais dans un fichier séparé, `personnality_prompt.txt`.

#### Corrections

- L'IA ne poursuivait pas ses messages quand ils contenaient un saut de ligne.
- Divers autres bugs.

</details>
