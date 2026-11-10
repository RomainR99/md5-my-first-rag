# Review

## Sommaire

- [`Agent.read_file` est une méthode statique](#read-file)
- [`{{CHUNKS}}` dans le prompt système](#chunks)
- [D’où vient `vector_db_path`](#vector-db-path)
- [`super().__init__()` crée `self.client`](#super-init)
- [`encode`, `batch_size=64` et `tolist()`](#embeddings)
- [`VectorDB.__init__` : ouvrir ou créer la base](#vector-db-init)
- [`load_vector_db` et `PersistentClient`](#load-vector-db)
- [`create_vector_db` : préparer la collection](#create-vector-db)
- [`collection.metadata.keys()`](#metadata-keys)
- [D'où vient le découpage de `chuncks`](#chuncks-csv)
- [`retrieve` : les passages les plus proches](#retrieve)
- [`retrieve` dans `build_context`](#build-context-retrieve)
- [L'affichage de la réponse et des passages](#affichage)
- [`rag_object = Rag(vector_db_path=vector_db_path)`](#rag-object)
- [`normalize_embeddings=True`](#normalize-embeddings)
- [Un `chunck` n'est pas 500 tokens](#chunck-taille)
- [L'embedding de la question et les 512 nombres](#embedding-question)

<a id="read-file"></a>

## `Agent.read_file` est une méthode statique

Dans `agent.py`, `read_file` est décorée avec `@staticmethod` parce qu’elle n’utilise rien de l’objet `Agent`.

Une méthode normale reçoit toujours l’instance en premier argument, `self`. Ici, le seul paramètre utile est `file_path` : ouvrir le fichier et renvoyer son texte. `self.client` (le client Groq créé dans `__init__`) n’intervient pas.

`@staticmethod` dit donc : cette fonction est rangée dans la classe `Agent`, mais elle ne dépend ni d’une instance ni de la classe. On peut l’appeler avec `Agent.read_file("rag_prompt_system.txt")` sans construire d’`Agent`.

Si le décorateur n’était pas là, Python passerait quand même l’instance en premier argument. `file_path` recevrait l’objet `Agent`, et `open(...)` planterait.

<a id="chunks"></a>

## `{{CHUNKS}}` dans le prompt système

Dans `rag_prompt_system.txt`, `{{CHUNKS}}` est un emplacement vide. Le fichier décrit les règles de l’assistant, puis laisse une zone « Contexte » que le code remplit juste avant d’appeler le modèle.

Dans `rag.py`, `build_context` récupère les extraits pertinents pour la question, lit ce fichier, puis remplace `{{CHUNKS}}` par ces extraits, un par ligne :

```python
prompt_system = Rag.read_file("./rag_prompt_system.txt")
prompt_system = prompt_system.replace("{{CHUNKS}}", "\n\t".join(documents))
```

Le texte obtenu devient le message `system` envoyé au modèle. La question de l’utilisateur part à part, dans le message `user`. Le modèle voit donc d’abord les règles, puis uniquement les passages retrouvés dans la base vectorielle, et il doit répondre à partir de ça.

<a id="vector-db-path"></a>

## D’où vient `vector_db_path`

Ces deux lignes ne s’exécutent que si le fichier est lancé directement, avec `python rag.py`. Elles fabriquent l’objet qui va répondre aux questions.

```python
vector_db_path = "my_vector_db"
rag_object = Rag(vector_db_path=vector_db_path)
```

La première ligne range un texte dans une variable. `my_vector_db` est le nom du dossier, à côté de `rag.py`, où Chroma a enregistré les extraits et leurs vecteurs.

La ligne suivante crée un `Rag` et lui donne ce nom de dossier. Les deux `vector_db_path` ne jouent pas le même rôle. À gauche du `=`, c’est le nom du paramètre attendu par `__init__`. À droite, c’est la variable définie juste au-dessus, donc le texte `"my_vector_db"`.

Ce texte arrive dans le constructeur, puis est passé tel quel à `VectorDB`. `VectorDB` ouvre le dossier `my_vector_db` s’il existe déjà sur le disque. `Rag` ne passe pas de corpus : au moment d’une question, ce dossier doit déjà avoir été construit.

<a id="super-init"></a>

## `super().__init__()` crée `self.client`

`super().__init__()` appelle le constructeur de la classe parente, `Agent`.

`Rag` hérite d'`Agent` (`class Rag(Agent)`). Comme `Rag` définit son propre `__init__`, Python n'exécute pas celui d'`Agent` tout seul. La ligne `super().__init__()` le déclenche. Dans `agent.py`, ce constructeur fait une seule chose : créer `self.client`, le client Groq, avec la clé d'API.

Ensuite, la ligne suivante ajoute ce qui est propre à `Rag` : `self.vector_db_object`, la base vectorielle ouverte avec le chemin reçu.

Après ces deux lignes, l'objet a donc les deux pièces dont `ask_rag` a besoin. `self.client` vient d'`Agent`, et `self.vector_db_object` vient de `Rag`. Sans `super().__init__()`, `self.client` n'existerait pas, et l'appel au modèle planterait.

Il n'y a pas de `self.agent`. `Agent` est le nom de la classe parente. `Rag` en hérite, donc l'objet construit avec `Rag(...)` est lui-même un `Agent`. Il ne contient pas un autre objet rangé dans `self.agent`.

`super().__init__()` exécute le constructeur d'`Agent` sur cet objet. Ce constructeur écrit `self.client = Groq(api_key=GROQ_API_KEY)`. `self.client` est le client de l'API Groq, posé directement sur le `Rag`. C'est lui que `ask_rag` utilise avec `self.client.chat.completions.create(...)`.

<a id="embeddings"></a>

## `encode`, `batch_size=64` et `tolist()`

Dans `get_embeddings`, `batch_size=64` dit à `encode` de calculer les vecteurs par groupes de 64 textes. `tolist()` transforme le tableau renvoyé par cette fonction en liste Python.

`encode` vient de `SentenceTransformer` parce que l'objet sur lequel il est appelé est construit avec cette classe. Le fichier l'importe, puis crée l'objet :

```python
from sentence_transformers import SentenceTransformer
self.sentence_transformers_object = SentenceTransformer(EMBEDDING_MODEL)
```

L'appel utilise cet objet : `self.sentence_transformers_object.encode(...).tolist()`. La méthode appartient donc à la classe de `self.sentence_transformers_object`.

`encode` ne vectorise pas tout le corpus d'un coup. Il prend 64 extraits, calcule leurs vecteurs, puis passe aux 64 suivants. Le nombre 64 n'est pas lu dans les données. C'est un choix écrit dans le code. La valeur par défaut de la bibliothèque est 32. Un lot plus grand va plus vite et occupe plus de mémoire. Le vecteur de chaque texte reste le même.

`tolist()` vient de NumPy. Ce n'est pas une fonction de ChromaDB. L'expression se lit de gauche à droite : `encode(...)` s'exécute d'abord et, avec ses réglages par défaut, renvoie un tableau NumPy. `.tolist()` est une méthode de ce tableau (`numpy.ndarray`) : elle le convertit en liste de listes de nombres. ChromaDB reçoit seulement le résultat, plus tard, dans `collection.add(embeddings=embeddings)`. `retrieve` réutilise la même liste pour la question.

<a id="vector-db-init"></a>

## `VectorDB.__init__` : ouvrir ou créer la base

Le constructeur de `VectorDB` choisit entre ouvrir une base déjà sur le disque et en fabriquer une nouvelle.

Il reçoit deux arguments. `vector_db_path` est le chemin du dossier Chroma, par défaut une chaîne vide. `corpus_df` est un tableau pandas du corpus, par défaut `None`.

`os.path.exists(vector_db_path)` regarde si ce dossier existe déjà. S'il existe, `load_vector_db` l'ouvre et relit le modèle d'embedding enregistré dans la collection `rag_knowledge`. Le corpus passé en argument n'est alors pas utilisé : le dossier déjà présent gagne.

Si le dossier n'existe pas, la deuxième branche regarde si `corpus_df` contient des lignes (`not corpus_df.empty`). Dans ce cas, `create_vector_db` crée le dossier, vectorise les textes et les enregistre.

Depuis `rag.py`, l'appel est `VectorDB(vector_db_path=vector_db_path)` : seul le chemin est fourni, `corpus_df` reste `None`. Ça suppose que le dossier `my_vector_db` existe déjà. S'il n'existe pas, `corpus_df.empty` plante, parce que `None` n'a pas d'attribut `empty`. La création à partir du CSV se fait dans `vector_db.py`, quand le fichier est lancé directement : il lit `05_corpus_rag.csv` et passe ce tableau au constructeur.

<a id="load-vector-db"></a>

## `load_vector_db` et `PersistentClient`

`load_vector_db` rouvre une base Chroma déjà enregistrée sur le disque et recharge le modèle qui a servi à calculer ses vecteurs.

`chromadb.PersistentClient(path=vector_db_path)` ouvre le dossier, par exemple `my_vector_db`. « Persistent » veut dire que les vecteurs sont lus dans les fichiers de ce dossier. Le client est gardé dans `self.chroma_vector_db` pour les recherches suivantes.

`PersistentClient` est déjà écrite. C'est une classe de la bibliothèque ChromaDB, installée via `chromadb` dans `requirements.txt`. Elle n'est pas dans ce dépôt. Le fichier commence par `import chromadb`, et `chromadb.PersistentClient(...)` crée un objet de cette classe. Le point signifie : prends `PersistentClient` à l'intérieur du module `chromadb`. Cet objet sait ouvrir le dossier de la base et y lire ou y écrire les vecteurs.

`get_collection(name="rag_knowledge")` ouvre la collection créée au moment de la construction de la base. Cette variable `collection` sert ici uniquement à lire ses métadonnées. Elle n'est pas conservée sur l'objet.

Lors de la création, la collection a été enregistrée avec une métadonnée `embedding_model`, le nom du modèle SentenceTransformer, ici `distiluse-base-multilingual-cased-v2`. Le `if` vérifie que cette clé est présente, puis charge ce même modèle dans `self.sentence_transformers_object`. Les questions devront être vectorisées avec lui : un autre modèle produirait des vecteurs qui ne se comparent pas à ceux déjà stockés.

Si la clé est absente, le `else` lève une exception et s'arrête. Sans le nom du modèle, le code ne saurait pas comment transformer une question en vecteur.

<a id="create-vector-db"></a>

## `create_vector_db` : préparer la collection

Les lignes 29 à 40 préparent la collection. Elles chargent le modèle d'embedding et créent la collection Chroma. Les textes du corpus sont ajoutés juste après, à partir de la ligne 42.

`SentenceTransformer(EMBEDDING_MODEL)` charge le modèle nommé dans `config.py`, `distiluse-base-multilingual-cased-v2`. L'objet est gardé dans `self.sentence_transformers_object`. C'est lui que `get_embeddings` utilisera pour transformer chaque extrait en vecteur.

`chromadb.PersistentClient(path=vector_db_path)` ouvre, ou crée, le dossier de la base, par exemple `my_vector_db`. Le client est gardé dans `self.chroma_vector_db`.

`get_or_create_collection` ouvre la collection `rag_knowledge` si elle existe déjà, et la crée sinon. Au premier passage, le dossier vient d'être créé, donc la collection est neuve. Le dictionnaire `metadata` y enregistre `embedding_model` avec le même nom de modèle. Plus tard, `load_vector_db` lit cette clé pour recharger exactement ce modèle.

Le mot « vide » n'est écrit nulle part. C'est une déduction à partir de l'ordre du code.

`create_vector_db` n'est appelée que si le dossier n'existe pas encore : dans `__init__`, la branche `os.path.exists(vector_db_path)` a échoué. `PersistentClient` crée donc un dossier neuf. À cet instant, il n'y a aucune collection dedans.

`get_or_create_collection` crée ensuite le conteneur `rag_knowledge`, avec seulement son nom et la métadonnée `embedding_model`. Ces lignes 29 à 40 ne passent ni textes, ni vecteurs, ni identifiants.

Le remplissage arrive après, à partir de la ligne 42 : les textes sont lus dans `corpus_df`, vectorisés, puis envoyés avec `collection.add(...)`. Avant cet `add`, la collection existe et ne contient aucun document.

<a id="metadata-keys"></a>

## `collection.metadata.keys()`

`keys()` porte sur le dictionnaire `collection.metadata`. C'est une méthode native de Python, pas de ChromaDB.

`collection` est un objet Chroma, renvoyé par `get_collection`. Chroma range les métadonnées de la collection dans `metadata`, sous forme d'un dictionnaire Python. Dans ce projet, à la création, ce dictionnaire vaut `{"embedding_model": "distiluse-base-multilingual-cased-v2"}`.

Sur un dictionnaire, `.keys()` donne les noms des entrées. Ici, la seule clé est `"embedding_model"`. Le test `"embedding_model" in collection.metadata.keys()` demande donc si ce nom est présent avant de charger le modèle.

Ce dictionnaire est écrit dans le code au moment de la création de la base. Il n'a pas été lu dans les fichiers de Chroma sur le disque.

La valeur du modèle est une constante dans `config.py` :

```python
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
```

`create_vector_db` importe cette constante et la range dans les métadonnées de la collection :

```python
collection = self.chroma_vector_db.get_or_create_collection(
    name="rag_knowledge",
    metadata={
        "embedding_model": EMBEDDING_MODEL
    }
)
```

`EMBEDDING_MODEL` est donc remplacé par le texte de `config.py`. La collection est enregistrée avec `{"embedding_model": "distiluse-base-multilingual-cased-v2"}`. Plus tard, `load_vector_db` relit `collection.metadata["embedding_model"]` et retrouve ce même texte.

<a id="chuncks-csv"></a>

## D'où vient le découpage de `chuncks`

Ce code ne découpe rien. Chaque élément de `chuncks` est déjà une ligne du CSV.

```python
chuncks = list(corpus_df["text"].values)
```

Cette ligne prend la colonne `text` de `05_corpus_rag.csv` et en fait une liste. Une ligne du fichier égale un élément de la liste. La première est « Le chat bleu de Bob s'appelle Henri. », avec l'identifiant `chunk_001`. La deuxième est la phrase sur Henri et le réfrigérateur, `chunk_002`. Le découpage a été fait avant, en écrivant le CSV : une phrase par ligne. Il n'y a dans ce projet ni taille de chunk, ni chevauchement, ni fonction qui coupe un long texte.

`batch_size=64` ne change pas ce découpage. Il dit seulement à `encode` de calculer les vecteurs 64 lignes à la fois. Les phrases restent celles du fichier.

<a id="retrieve"></a>

## `retrieve` : les passages les plus proches

`retrieve` transforme la question en vecteur, puis demande à Chroma les `n` passages les plus proches. Par défaut, `n` vaut 3.

`get_embeddings([question])` calcule le vecteur de la question avec le même modèle que les lignes du CSV. `get_collection("rag_knowledge")` redemande la collection. `query` compare le vecteur de la question à tous les vecteurs stockés et garde les `n` plus proches. `n_results=n` fixe ce nombre.

Chroma renvoie un dictionnaire. Comme une seule question a été envoyée, chaque entrée est une liste dont le premier élément correspond à cette question. `results["documents"][0]` est la liste des textes retenus. `results["metadatas"][0]` est la liste de leurs métadonnées, `source` et `category`, dans le même ordre.

Ces deux listes sont ce que `build_context` récupère dans `rag.py`. Les textes sont ensuite collés à la place de `{{CHUNKS}}`.

`get_embeddings` n'écrit nulle part « je veux une liste ». Son paramètre s'appelle seulement `chuncks`, et il le passe tel quel à `encode`. On le voit aux deux endroits qui l'appellent. À la création, l'argument est `list(corpus_df["text"].values)` : la colonne `text` transformée en liste de phrases. Dans `retrieve`, la question est mise entre crochets, `[question]`. Si une chaîne seule suffisait, ce crochet serait inutile. Il force une liste d'un seul texte, pour que `encode` renvoie un lot d'un vecteur, puis `tolist()` une liste de listes. C'est cette forme que `query` attend dans `query_embeddings`.

`get_collection("rag_knowledge")` ne remplit rien. `get_collection` demande à Chroma une collection qui existe déjà sous ce nom. Le remplissage a eu lieu plus tôt, dans `create_vector_db` : `get_or_create_collection(name="rag_knowledge", ...)` crée cette collection, puis `collection.add(...)` y écrit les textes et les vecteurs dans le dossier ouvert par `PersistentClient`. `retrieve` réutilise le même client, `self.chroma_vector_db`, et redemande la collection du même nom. Les documents sont ceux déjà enregistrés sur le disque.

<a id="build-context-retrieve"></a>

## `retrieve` dans `build_context`

À la ligne 14 de `rag.py`, `retrieve` cherche dans la base les passages proches de `question`, puis `build_context` les colle dans le prompt.

`self.vector_db_object` est le `VectorDB` créé dans `__init__` avec le dossier `my_vector_db`. L'appel ne passe que la question. Le paramètre `n` de `retrieve` garde donc sa valeur par défaut, 3 : trois textes, trois métadonnées.

`retrieve` renvoie deux listes, dans cet ordre : les textes, puis leurs métadonnées. L'affectation `documents, metadatas = ...` range la première dans `documents` et la seconde dans `metadatas`.

Seule `documents` sert dans cette fonction. `"\n\t".join(documents)` recolle les trois phrases, séparées par un saut de ligne et une tabulation, et ce bloc remplace `{{CHUNKS}}` dans le fichier de prompt. `metadatas` n'est pas lue ici. Elle est renvoyée avec le prompt et les textes, pour que `ask_rag` puisse les afficher ensuite.

`retrieve` n'est pas une fonction native de Python. C'est une méthode écrite dans ce projet, sur la classe `VectorDB`, dans `vector_db.py`.

Python ne connaît pas ce nom. Il exécute la fonction parce que `self.vector_db_object` est un objet `VectorDB`, et que cette classe contient `def retrieve(self, question, n=3)`. L'appel `self.vector_db_object.retrieve(question)` veut dire : prends cet objet et lance sa méthode `retrieve`.

À l'intérieur, ce sont d'autres appels qui font le travail : `get_embeddings`, écrit dans le même fichier, et `collection.query`, qui vient de ChromaDB.

<a id="affichage"></a>

## L'affichage de la réponse et des passages

Les lignes 52 à 59 de `rag.py` affichent la réponse du modèle, puis les trois passages qui ont servi à la produire. Elles ne tournent que lorsque le fichier est lancé avec `python rag.py`.

`rag_response` est le texte renvoyé par Groq. `print("-"*20)` écrit une ligne de 20 tirets pour séparer cette réponse de la suite.

`documents` et `metadatas` ont le même nombre d'éléments, dans le même ordre : le texte n° 0 va avec les métadonnées n° 0. La boucle parcourt les indices `0`, `1` et `2`. Pour chacun, elle affiche le numéro, le texte du passage, puis chaque entrée du dictionnaire de métadonnées. `items()` donne les paires de ce dictionnaire, ici `source` et `category`. Une ligne `---` ferme le passage avant le suivant.

<a id="rag-object"></a>

## `rag_object = Rag(vector_db_path=vector_db_path)`

Cette ligne fabrique un objet `Rag` et le range dans la variable `rag_object`.

`Rag(...)` appelle le constructeur `__init__`. Le texte entre parenthèses est l'argument qu'on lui donne. Les deux `vector_db_path` ne jouent pas le même rôle. À gauche du `=`, c'est le nom du paramètre attendu par `__init__`. À droite, c'est la variable de la ligne du dessus, donc le texte `"my_vector_db"`.

Dans `__init__`, ce texte est passé à `VectorDB`, qui ouvre le dossier de la base. `super().__init__()` crée aussi `self.client`, le client Groq. L'objet obtenu a donc les deux pièces nécessaires. Il est gardé dans `rag_object` pour pouvoir appeler `rag_object.ask_rag(...)` juste après.

<a id="normalize-embeddings"></a>

## `normalize_embeddings=True`

`normalize_embeddings=True` ramène chaque vecteur à une longueur de 1. La comparaison porte alors sur la direction du vecteur, pas sur sa taille.

`encode` produit 512 nombres. Sans normalisation, un texte plus long peut donner un vecteur plus grand. Deux phrases proches par le sens, mais de longueurs différentes, peuvent alors paraître éloignées simplement à cause de cette taille.

Avec `True`, chaque vecteur est divisé par sa longueur. Deux vecteurs de longueur 1 : leur proximité mesure l'angle entre eux, c'est-à-dire la similarité de sens. Les textes du CSV et la question passent tous les deux par `get_embeddings`, donc ils sont normalisés de la même façon avant que Chroma ne cherche les plus proches.

<a id="chunck-taille"></a>

## Un `chunck` n'est pas 500 tokens

Ici, un `chunck` n'est pas un bloc de 500 tokens. C'est une ligne de la colonne `text` du CSV, déjà écrite comme une phrase.

Le fichier contient 200 lignes. La plus courte fait 7 mots, la plus longue 22, soit au plus 117 caractères. Ça représente quelques dizaines de tokens, pas 500.

`chuncks` dans `get_embeddings` est simplement la liste de ces 200 phrases. Le code ne coupe aucun texte et ne compte aucun token. Une découpe à 500 tokens est une méthode courante dans d'autres projets RAG, quand on part d'un long document. Elle n'est pas utilisée ici.

<a id="embedding-question"></a>

## L'embedding de la question et les 512 nombres

La question reste une phrase jusqu'à `retrieve`. C'est là qu'elle devient un vecteur, avec le même modèle que les lignes du CSV.

Dans `rag.py`, le texte « Quelle est la couleur et le nom du chat de Bob ? » est passé à `ask_rag`, puis à `build_context`, puis à `VectorDB.retrieve`. Aucune de ces étapes ne calcule encore de vecteur : elles se transmettent la phrase.

Dans `retrieve`, la phrase est mise seule dans une liste, `[question]`, et cette liste est donnée à `get_embeddings`. Cette méthode appelle `encode` sur `self.sentence_transformers_object`. Cet objet est le modèle `distiluse-base-multilingual-cased-v2`, rechargé au démarrage depuis la métadonnée `embedding_model` de la collection.

`encode` transforme la phrase en 512 nombres, puis `normalize_embeddings=True` ramène ce vecteur à une longueur de 1. `tolist()` le convertit en liste Python. Comme la liste d'entrée ne contenait qu'une phrase, le résultat est une liste qui contient un seul vecteur.

Ce vecteur est ensuite passé à `collection.query`. Chroma le compare aux vecteurs déjà stockés et renvoie les trois passages les plus proches. Le modèle Groq ne voit jamais ce vecteur : il ne reçoit que le texte de ces trois passages.

512 n'est pas un réglage de ce projet. C'est la taille de sortie du modèle `distiluse-base-multilingual-cased-v2`.

Chaque phrase doit devenir un vecteur de la même longueur, sinon Chroma ne peut pas comparer la question aux lignes du CSV. Les auteurs du modèle ont fixé cette longueur à 512 au moment de l'entraînement. `encode` renvoie donc toujours 512 nombres, que la phrase fasse 7 mots ou 22.

Ce nombre vient du Universal Sentence Encoder, dont ce modèle est une version distillée. L'encodeur d'origine produit des vecteurs de 512. Le réseau chargé ici, le module `2_Dense` téléchargé au premier lancement, compresse la représentation interne du texte jusqu'à cette taille. 512 est un compromis choisi par eux : assez grand pour distinguer les sens, assez petit pour stocker et comparer les vecteurs rapidement. Une autre taille demanderait un autre modèle déjà entraîné.
