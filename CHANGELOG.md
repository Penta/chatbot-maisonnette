# Changelog — Chatbot Discord

Les versions les plus récentes sont en haut. Chaque entrée commence par un résumé en gras, suivi du détail.

**Versions :** [5.14](#version-514) · [5.13](#version-513) · [5.12](#version-512) · [5.11](#version-511) · [5.10](#version-510) · [5.9](#version-59) · [5.8](#version-58) · [5.7](#version-57) · [5.6](#version-56) · [5.5](#version-55) · [5.4.0](#version-540) · [5.3.0](#version-530) · [5.2.0](#version-520) · [5.1.0](#version-510-1) · [5.0.0](#version-500) · [Avant la 5.0](#avant-la-version-50)

---

## Version 5.14

### Changements

- **Désactiver la mémoire ne touche plus au fichier général.** Rika a pour consigne de ne rien y noter sur les membres : seule la fiche est effacée, et le nom retiré des sources des autres fiches (aussi sous sa forme simple pour un pseudo décoré comme « 𝓑𝓸𝓫 »). `/memoire` invite le membre à faire vérifier le fichier général par un administrateur.
- **Une source est gardée par l'identifiant Discord du membre.** « d'après Alice » désigne toujours la bonne Alice, même si un autre membre prend son pseudo, et affiche son nom actuel. Désactiver la mémoire retire exactement ses mentions. Les notes plus anciennes restent écrites avec le nom.
- **Les fiches ne se modifient plus à la main, ligne par ligne seulement.** Dans `/memoire`, on coche les lignes à supprimer dans un menu, et le membre écrit ses surnoms avec le bouton « Surnoms ». Dans `/memoire-serveur`, l'administrateur peut aussi ajouter des lignes. Rika garde au plus 25 lignes par fichier ; une fiche plus longue n'est pas tronquée.
- **Les versions des dépendances sont figées.** `requirements.txt` indique les versions exactes testées : une nouvelle version d'une bibliothèque n'arrive plus sans changement du dépôt. Pour une installation existante, relancer `pip install -r requirements.txt`.

### Corrections

- **Rika ne peut plus vider une fiche ou le fichier général sur simple demande.** Un membre pouvait lui faire réécrire la fiche d'un autre avec une seule ligne, et les anciennes notes étaient perdues. Elle ne peut plus en retirer plus de la moitié d'un coup ; effacer se fait avec `/memoire` ou `/memoire-serveur`.

- **Désactiver la mémoire n'efface plus les sources d'un autre membre.** Un membre qui prenait le pseudo d'un autre avant de désactiver sa mémoire faisait remplacer par « un membre » le nom de cet autre dans toutes les fiches. Un nom porté par deux membres est maintenant laissé, et noté dans le log.

- **Rika ne joue plus à un jeu Steam pas encore sorti.** Les jeux annoncés sur la boutique sont maintenant écartés du statut « joue à », comme les jeux pour adultes. Rien à régler.

- **Une « version » invalide dans `registre.json` n'empêche plus le bot de démarrer.** Le bot démarre avec la mémoire des membres coupée et le log indique quoi corriger : `version` doit être un nombre entier, sans guillemets. Corrige le fichier puis redémarre.

---

## Version 5.13

### Corrections

- **Une image aux dimensions énormes ne fait plus tomber le bot.** Un fichier léger mais de dizaines de milliers de pixels de côté pouvait occuper plus d'1 Go de mémoire une fois décodé, et faire redémarrer le bot. Il est maintenant refusé d'emblée au-delà de 40 millions de pixels (120 millions pour la plupart des photos JPEG), et les images sont préparées une à la fois.

- **Une coupure réseau pendant un téléchargement ne fait plus ignorer le message.** Si une pièce jointe ne se télécharge pas (connexion coupée, ou plus de 30 secondes), Rika demande de réessayer ; une image illisible dans un message cité ou transféré est simplement ignorée. Avant, le message restait sans réponse.

- **Plusieurs gros fichiers envoyés pendant que Rika écrit ne font plus échouer sa réponse.** Ils partaient tous dans un seul appel à Claude, qui pouvait dépasser ses limites. Quand leurs PDF et images sont trop lourds ensemble (plus de 18 Mo, ou plus de 100 pages et images), Rika répond maintenant en plusieurs fois, dans l'ordre, chaque réponse citant le message auquel elle répond.

- **Les PDF protégés seulement contre la modification sont lus.** Un PDF chiffré en AES sans mot de passe d'ouverture (courant avec Acrobat ou LibreOffice) était refusé comme « illisible ». Il est maintenant lu, grâce à la nouvelle dépendance `cryptography` installée avec les autres.

- **Les grandes images en gris 16 bits ne deviennent plus toutes blanches.** Un scan, une radio ou une photo d'astronomie en PNG ou TIFF 16 bits (ou 32 bits), une fois réduit, arrivait tout blanc ou tout noir chez Claude. Ses gris sont maintenant ramenés sur 8 bits sans être écrêtés (Pillow 11 ou plus, installé avec les autres dépendances).

- **Un octet abîmé ne déforme plus tous les accents d'un fichier texte.** Dans un `.txt`, `.log` ou `.csv` en UTF-8, un seul octet invalide faisait lire tout le fichier en Windows-1252 (« CafÃ© »). Seul cet octet est maintenant relu ainsi, si le fichier est surtout en UTF-8 ; les fichiers « ANSI » restent bien lus.

- **Le journal dit pourquoi un bot qui mentionne Rika est ignoré.** Quand un autre bot écrit @Rika, une ligne indique si Discord a bien transmis la mention et si le message est pris en compte ou ignoré (mention absente, ou limite de réponses aux bots d'affilée).

---

## Version 5.12

> **En bref :** plusieurs images par message, des réponses groupées aux mentions, et un bot qui vérifie ses réglages au démarrage au lieu de tomber en panne plus tard.

### Changements

- **Journal réglable.** `LOG_LEVEL` choisit le détail du journal (`DEBUG` pour enquêter, `INFO` par défaut), et `LOG_FILE=off` n'écrit que sur la console, pratique en conteneur. Le démarrage rappelle aussi les principaux réglages et les chemins des fichiers.

- **Plusieurs images dans un même message.** Rika lit jusqu'à 4 images à la fois, dans l'ordre, y compris dans un message cité ou transféré, et peut les comparer ou parler de « la deuxième ». Si une image pose problème, le message d'erreur dit laquelle. Un PDF reste seul, sans image.

- **Un message de démarrage plus soigné.** Le message « En ligne » affiche l'avatar du bot, sa version, le modèle Claude utilisé (« Sonnet 5.5 ») et l'heure du démarrage.

### Corrections

- **Deux bots ne tournent plus en boucle.** Face à un bot qui lui répond à chaque fois, Rika pouvait enchaîner les réponses, et les appels à Claude, sans fin. Elle s'arrête maintenant après 5 messages de bots d'affilée dans un salon, jusqu'au prochain message d'un membre ou après 10 minutes de calme.

- **Une rafale de mentions ne lance plus un appel par message.** Comme dans son salon, les mentions envoyées pendant que Rika écrit, ou pendant une absence, reçoivent une seule réponse groupée, qui tient compte de celle qu'elle vient de poster : 10 « @Rika ? » d'affilée coûtaient 10 appels à Claude, ils en coûtent maintenant deux.

- **Erreurs de démarrage plus claires.** Toutes les erreurs du `.env` sont listées d'un coup, au lieu d'une par redémarrage, et elles sont aussi écrites dans `bot.log`. Sans Internet, ou si Discord est en panne, un message clair remplace la trace technique.

- **Un `.env` entre guillemets marche aussi sous Docker.** `DISCORD_TOKEN="abc"` donnait « Token Discord invalide » avec `docker run --env-file` : le bot retire maintenant lui-même les guillemets. Le README explique les autres différences de `--env-file`.

- **« Réflexion : off » n'est plus affiché à tort.** Sur Opus 5.5, dont la réflexion ne peut pas être coupée, le journal de démarrage indiquait « off » avec `THINKING=off` ; il dit maintenant que c'est impossible sur ce modèle.

- **Documentation des réglages alignée.** `.env.exemple` et le README décrivent maintenant exactement ce que fait le bot pour `THINKING`, `EFFORT` et `MAX_TOKENS`, avec les mêmes valeurs proposées.

- **« Modifier » ne signale plus de faux conflit.** Quand l'enregistrement était lent, la fenêtre de `/memoire` restait ouverte alors que le texte était sauvé, et un second clic disait à tort que le fichier avait changé entre-temps. La fenêtre se ferme maintenant dès l'envoi.

- **Une réponse trop lente n'est plus relancée trois fois.** Un membre pouvait attendre 4 minutes, recherches web refacturées à chaque essai, avant « Je mets trop de temps à répondre ». `API_TIMEOUT` borne maintenant l'attente totale, et le bot prévient au démarrage s'il est trop court pour `MAX_TOKENS`.

- **`MAX_TOKENS` vérifié en tenant compte de la réflexion.** Avec `MEMORY=on` et un effort élevé, une note de mémoire pouvait être perdue sans avertissement, faute de place. Le bot refuse maintenant de démarrer sous le minimum et prévient dans le message « En ligne » sous la valeur conseillée ; `.env.exemple` propose `MAX_TOKENS=6000` et `API_TIMEOUT=90`.

- **Une personnalité illisible ne rend plus Rika muette.** Un fichier de personnalité illisible (pas en UTF-8, ou un dossier au lieu d'un fichier) faisait échouer chaque message. Rika garde maintenant la dernière personnalité valide, et le journal explique comment corriger le fichier.

- **Les réglages ont un plafond.** Une valeur trop grande ne se voyait qu'au premier message, ou coûtait cher sans prévenir : `CONTEXT_MESSAGE_LIMIT=2000` pouvait coûter plusieurs dizaines de centimes par mention. Au-delà du plafond (100 pour `CONTEXT_MESSAGE_LIMIT`, les autres dans `.env.exemple`), le bot refuse maintenant de démarrer, avec un message clair.

- **`PRESENCE_WEIGHTS` mieux vérifié.** Une virgule en trop est ignorée. Un statut écrit deux fois, ou `jeu` comme seul statut possible sans `STEAM_API_KEY`, arrête maintenant le bot avec un message clair.

- **Un dossier absent est signalé au démarrage.** Si le dossier de `LOG_FILE`, `HISTORY_FILE` ou `MEMORY_DIR` n'existe pas ou n'est pas accessible en écriture (volume Docker mal monté, faute de frappe), le bot refuse de démarrer au lieu de tourner sans rien sauvegarder. L'historique résiste aussi mieux à une coupure de courant.

- **Un `CHANNEL_ID` faux arrête le bot.** Avec un mauvais salon ou une permission manquante, le bot restait connecté sans répondre nulle part. Il vérifie maintenant le salon au démarrage et s'arrête en disant quoi corriger ; si Discord est seulement en panne, il patiente.

- **Une clé Anthropic ou un modèle faux arrête le bot.** Une `ANTHROPIC_API_KEY` ou un `CLAUDE_MODEL` invalide ne se voyait qu'au premier message, avec « Je suis mal configuré ». Le bot les vérifie maintenant au démarrage ; seul un crédit épuisé ne se voit toujours qu'au premier message.

- **Une longue réponse avec du code n'est plus coupée.** Quand un bloc de code commençait juste à la fin d'un morceau de 2000 caractères, ce morceau dépassait la limite de Discord et la suite de la réponse n'était pas envoyée. Le bloc passe maintenant au morceau suivant ; même chose pour les fiches de `/memoire`. Un bloc qui se termine juste à la fin d'un morceau n'affiche plus de bloc vide au début du suivant.

- **Une date sans fuseau dans `presence_state.json` n'empêche plus le démarrage.** Après une modification à la main, une date comme `2026-11-03T10:00:00` arrêtait le bot dès le lancement. Le fichier est maintenant ignoré, avec un avertissement dans le journal, et Rika tire un nouveau statut.

- **Une ligne de code très longue ne fige plus le bot.** Une réponse dont un bloc de code s'ouvrait sur une ligne de près de 2000 caractères (« ```py » suivi de texte) figeait tout le bot, dans tous les salons, jusqu'au redémarrage. Elle est maintenant découpée normalement.

---

## Version 5.11

> **En bref :** la mémoire des membres devient un choix de chacun (`/memoire`), elle est beaucoup mieux protégée contre les abus, et elle respecte davantage la vie privée.

### Changements

- **La mémoire des membres s'active soi-même.** Elle est désactivée par défaut, et chaque membre qui le souhaite l'active avec `/memoire`, après avoir lu ce qu'elle retient et qui peut voir sa fiche. Le message de bienvenue, envoyé en message privé au premier message de chacun, disparaît. Rien n'est noté sur un membre qui ne l'a pas activée, même quand les autres parlent de lui.

  Au premier démarrage, la mémoire reste activée pour les membres qui avaient déjà une fiche et ne l'avaient pas désactivée ; elle est désactivée pour les autres.

### Corrections — ce que Rika retient

- **Une info sur un membre ne va que dans sa fiche.** En désactivant la mémoire, un membre efface aussi les lignes du fichier général qui le citent.

- **Une fiche effacée reste effacée.** Une fiche effacée par « Désactiver la mémoire » ne peut plus être recréée par une note en cours ni par une fenêtre `/memoire` restée ouverte.

- **Ce qu'on retire de sa fiche n'y revient plus.** Avant, Rika le retrouvait dans la conversation et le notait de nouveau au message suivant. Désormais, une fiche ou le fichier général modifié (ou effacé) avec `/memoire` est montré à Rika avec la date de la modification : elle ne note plus ce qui a été dit avant et n'y figure pas. Ce qu'on lui redit ensuite se note normalement. Même chose après « Désactiver la mémoire », si le membre la réactive. Les dates sont gardées dans `memories/modifications.json`.

- **Les surnoms sont mieux reconnus.** La ligne « Surnoms : … » est trouvée même si elle n'est pas la première de la fiche (ajoutée à la fin, ou fiche modifiée à la main), avec ou sans puce ; plusieurs lignes « Surnoms » sont regroupées. Avant, elles étaient ignorées sans rien dire. « Surnom : … » au singulier et « **Surnoms :** … » en gras sont aussi reconnus.

- **Chemins plus souples.** L'outil accepte `memories/membres/…` (sans « / » au début), comme `/memories/membres/…`.

- **`MAX_TOKENS` passe à 4000 par défaut** (1500 avant), assez pour que Rika réécrive le fichier général quand il est plein. Avant, la réécriture était coupée et la note perdue, à chaque nouvel essai. Le bot prévient au démarrage si `MAX_TOKENS` est trop petit pour `MEMORY_SERVER_LIMIT`, et le log signale un appel d'outil coupé.

### Corrections — vie privée

- **Les données personnelles sont protégées.** Rika ne note plus les confidences ni ses jugements sur quelqu'un. Une information sensible (santé, sexualité, religion, opinions, origine, adresse, argent) ne se note que si le membre la donne lui-même, et elle ne l'évoque ensuite qu'avec lui. Elle se sert de ce qu'elle sait dans la conversation, mais ne récite plus la fiche d'un membre à quelqu'un d'autre. Les fiches déjà écrites ne sont pas modifiées.

- **Les membres savent qui peut lire leur fiche.** Ils sont prévenus que les administrateurs du serveur peuvent la voir et la corriger (avant d'activer la mémoire, et sous leur fiche dans `/memoire`). Chaque ouverture de la fiche d'un autre membre avec `/memoire` est notée dans le log, sans son contenu.

- **Les sources nommées sont annoncées.** Avant d'activer la mémoire, chacun est prévenu que les sources sont notées avec leur nom (« d'après Toto ») et visibles par la personne concernée : ce qu'on raconte sur un autre membre peut lui être montré dans sa fiche.

- **Pas de fiche écrite sans accord.** Un administrateur ne peut plus écrire la fiche d'un membre qui n'a pas activé la mémoire (« Modifier » grisé, avec une explication) ; il peut toujours l'effacer.

### Corrections — sécurité

- **Le bot ne sert plus que son propre serveur** (celui de `CHANNEL_ID`). Invité sur un autre serveur, il ignore les messages et refuse les commandes : un administrateur de ce serveur pouvait lire et réécrire la mémoire du vrai serveur avec `/memoire` et `/memoire-serveur`, et les réponses étaient payées par le propriétaire du bot. Les messages privés sont aussi ignorés. Le log signale au démarrage les autres serveurs où se trouve le bot, et le README explique comment décocher *Public Bot* et *User Install* dans le portail Discord.

- **Plus d'usurpation par le pseudo.** Rika voit chaque auteur avec son nom d'utilisateur Discord, unique (« Johnounet (johnk) »), quand il diffère de son pseudo, y compris pour l'auteur d'un message cité. Deux membres qui portent le même pseudo ne se confondent plus, et personne ne peut faire noter une info dans la fiche d'un autre en prenant son pseudo. Les fiches jointes indiquent celles des auteurs du message, et une fiche ne peut plus imiter l'en-tête d'une autre. Les messages déjà enregistrés gardent l'ancien format.

- **On ne peut plus prendre le nom d'un autre comme surnom.** Un surnom qui reprend le nom Discord d'un autre membre (pseudo, nom d'utilisateur ou ancien pseudo) est ignoré dans la liste des membres : Rika ne note plus dans sa fiche ce qu'on raconte sur l'autre.

- **Les notes restent des notes.** Le fichier général et la liste des membres sont présentés à Rika comme des notes, pas des consignes, et une note ne peut plus sortir de ce cadre, même avec une balise non fermée ou des chevrons pleine largeur. Rika ne note plus de règle qui s'appliquerait à tout le monde (façon d'écrire, sujet à éviter), même présentée comme votée ou venant d'un administrateur ; une préférence personnelle (« veut qu'on l'appelle Majesté ») se note toujours dans la fiche. Au plus 5 surnoms de 32 caractères par membre. Le texte écrit dans le fichier général apparaît dans le log, pour que l'administrateur puisse vérifier.

- **Un membre ne peut plus vider le fichier général** en désactivant sa mémoire. Seuls ses noms Discord servent à retrouver les lignes qui le citent, plus les surnoms de sa fiche, qu'il écrit lui-même. Si plus de 3 lignes le citent (pseudo qui est un mot courant, comme « Soleil »), aucune n'est retirée et le log demande à l'administrateur de vérifier avec `/memoire-serveur`.

### Corrections — `/memoire` et les commandes

- **`/memoire` ne promet plus d'effacer plus qu'il ne le fait.** Le bouton « Tout effacer » d'une fiche devient « Effacer la fiche », et la confirmation précise que ce qui a été dit récemment reste dans la conversation. « Désactiver la mémoire » ne dit plus que « plus rien ne sera retenu », et remplace maintenant le nom du membre par « un membre » là où il est cité comme source dans les fiches des autres (« d'après Bob », « selon Bob »).

- **« Modifier » n'écrase plus les changements récents.** La fenêtre n'enregistre plus rien si le fichier a changé depuis son ouverture (note de Rika, désactivation d'un membre qui retire des lignes du fichier général). Avant, elle réécrivait le fichier tel qu'il était à l'ouverture : la note était perdue, ou la ligne retirée revenait. Quand un texte n'est pas enregistré (fichier changé, trop long…), il est renvoyé en fichier joint pour ne pas être perdu.

- **« Modifier » respecte la limite du fichier général** : 5000 caractères (`MEMORY_SERVER_LIMIT`). Sa deuxième zone de texte laissait en écrire 8000, refusés ensuite à l'enregistrement.

- **Un fichier trop long s'ouvre de nouveau.** Un fichier plus long que sa limite (`MEMORY_*_LIMIT` baissée, fichier modifié sur le disque) s'ouvre en entier dans « Modifier », avec la limite rappelée pour le raccourcir. Avant, la fenêtre ne s'ouvrait pas (« Échec de l'interaction »). Au-delà de 20 000 caractères, trop pour une fenêtre Discord, son contenu est envoyé en fichier joint.

- **Des erreurs compréhensibles.** Une erreur dans une commande ou un bouton affiche « Une erreur est survenue » au lieu du seul « Échec de l'interaction » de Discord. Les boutons de `/memoire` se grisent au bout de 10 minutes, avec un message qui invite à relancer la commande, au lieu de rester affichés sans répondre. La fenêtre « Modifier » peut être validée pendant une heure, au lieu de 10 minutes.

- **La fiche d'un membre parti reste accessible** avec `/memoire membre:` (par son identifiant). Avant, la commande échouait, et lui-même ne pouvait plus la lancer : sa fiche restait pour toujours. Un administrateur peut aussi désactiver sa mémoire à sa place, ce qui efface tout comme s'il l'avait fait.

### Corrections — fiabilité et performances

- **La mémoire coûte moins cher sur un grand serveur.** Le prompt ne liste plus que les membres qui ont une fiche, au lieu de tous ceux que le bot a vus ou qu'on a mentionnés. Au plus 5 fiches sont jointes à un message (les auteurs d'abord, puis l'auteur du message cité, puis les membres mentionnés) ; les autres sont seulement nommées, et Rika peut les lire si besoin.

- **Le bot ne relit plus toutes les fiches à chaque message.** Elles sont lues une fois au démarrage, puis suivies à chaque note. Avec 500 membres, chaque message bloquait le bot près d'une seconde (mesuré sous Windows). Une fiche modifiée directement sur le disque n'est prise en compte dans la liste qu'au redémarrage.

- **Une fiche illisible ne bloque plus le bot.** Une fiche ou un fichier général illisible (enregistré à la main dans un autre encodage, problème de droits) est signalé une fois dans le log, affiché avec « � » à la place des caractères invalides, et n'est plus modifié tant qu'il n'est pas corrigé. Une fiche enregistrée avec BOM (Bloc-notes) garde sa ligne « Surnoms ».

- **Un registre abîmé coupe la mémoire au lieu de la réinitialiser.** Si `memories/registre.json` est abîmé ou mal modifié à la main, la mémoire des membres est coupée jusqu'à sa réparation, au lieu de repartir d'un registre vide qui réactivait la mémoire de ceux qui l'avaient désactivée. Le fichier n'est plus mis de côté ni réécrit ; le log et `/memoire` signalent la panne. Un registre corrigé à la main et enregistré avec BOM (Bloc-notes) reste lisible.

- **Un registre impossible à enregistrer est signalé.** Si le registre ne peut pas être enregistré (disque plein, droits), « Désactiver la mémoire » et « Réactiver la mémoire » le disent, au lieu d'annoncer un réglage qui serait perdu au redémarrage du bot. Le réglage s'applique quand même, et la prochaine sauvegarde réussie du registre l'enregistre.

### Corrections — démarrage et réponses

- **Un modèle inconnu est signalé dès le démarrage** (`CLAUDE_MODEL` mal écrit, par exemple), et plus seulement au premier message. Les messages d'erreur citent `CLAUDE_MODEL` (et non `MODEL`, une variable qui n'existe pas).

- **`TIMEZONE` mal formée** (« Europe/Paris/ », « ../Paris ») ou trop longue : message d'erreur clair au démarrage, au lieu d'une trace Python.

- **Plus de « […] » envoyé seul.** Une réponse coupée avant son premier mot n'envoie plus « […] » seul sur Discord (ni dans l'historique), mais un message d'explication.

### Divers

- **Secrets masqués.** Le token Discord et les clés Anthropic et Steam n'apparaissent plus quand la configuration est affichée (précaution : aucun log ne l'affiche aujourd'hui).

- **Premiers tests automatisés** (mémoire des membres) : lancer `python -m unittest` à la racine du projet.

- **Écritures plus sûres.** Les fiches et le registre des membres sont envoyés jusqu'au disque avant d'être remplacés : une coupure de courant ne laisse plus de fichier vide. Une écriture ratée ne laisse plus de fichier `.tmp` derrière elle.

---

## Version 5.10

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

> **En bref :** moins de réflexion qui fuit dans les réponses, et des logs plus propres.

### Corrections

- **Avertissements vocaux masqués.** Les avertissements de discord.py sur le vocal (PyNaCl et davey non installés) n'apparaissent plus à chaque démarrage : le bot n'utilise pas le vocal.

- **Réflexion affichée.** Avec Sonnet 5.5 et `EFFORT=high`, le modèle écrivait parfois sa réflexion dans sa réponse (faux appel d'outil `<invoke name="think">…`, raisonnement entre balises `<reasoning>…`), quand il choisissait de ne pas réfléchir. Ça n'arrive plus avec `EFFORT=medium` : c'est le réglage conseillé.
  - Les consignes ajoutées automatiquement à la personnalité suivent les recommandations d'Anthropic contre ces fuites : plus d'interdiction de commenter les recherches web (ce commentaire n'est de toute façon jamais affiché), et une consigne générale contre les balises internes.
  - Au démarrage, le log indique le modèle, la réflexion, l'effort et la recherche web réellement utilisés.

- **Logs moins bavards.** Chaque requête HTTP vers l'API d'Anthropic n'est plus journalisée (le SDK 2.x passe par httpx2, qui n'était pas rendu silencieux).

---

## Version 5.4.0

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
