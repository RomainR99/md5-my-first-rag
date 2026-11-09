# Review

## `Agent.read_file` est une méthode statique

Dans `agent.py`, `read_file` est décorée avec `@staticmethod` parce qu’elle n’utilise rien de l’objet `Agent`.

Une méthode normale reçoit toujours l’instance en premier argument, `self`. Ici, le seul paramètre utile est `file_path` : ouvrir le fichier et renvoyer son texte. `self.client` (le client Groq créé dans `__init__`) n’intervient pas.

`@staticmethod` dit donc : cette fonction est rangée dans la classe `Agent`, mais elle ne dépend ni d’une instance ni de la classe. On peut l’appeler avec `Agent.read_file("rag_prompt_system.txt")` sans construire d’`Agent`.

Si le décorateur n’était pas là, Python passerait quand même l’instance en premier argument. `file_path` recevrait l’objet `Agent`, et `open(...)` planterait.

## `{{CHUNKS}}` dans le prompt système

Dans `rag_prompt_system.txt`, `{{CHUNKS}}` est un emplacement vide. Le fichier décrit les règles de l’assistant, puis laisse une zone « Contexte » que le code remplit juste avant d’appeler le modèle.

Dans `rag.py`, `build_context` récupère les extraits pertinents pour la question, lit ce fichier, puis remplace `{{CHUNKS}}` par ces extraits, un par ligne :

```python
prompt_system = Rag.read_file("./rag_prompt_system.txt")
prompt_system = prompt_system.replace("{{CHUNKS}}", "\n\t".join(documents))
```

Le texte obtenu devient le message `system` envoyé au modèle. La question de l’utilisateur part à part, dans le message `user`. Le modèle voit donc d’abord les règles, puis uniquement les passages retrouvés dans la base vectorielle, et il doit répondre à partir de ça.

## D’où vient `vector_db_path`

Ces deux lignes ne s’exécutent que si le fichier est lancé directement, avec `python rag.py`. Elles fabriquent l’objet qui va répondre aux questions.

```python
vector_db_path = "my_vector_db"
rag_object = Rag(vector_db_path=vector_db_path)
```

La première ligne range un texte dans une variable. `my_vector_db` est le nom du dossier, à côté de `rag.py`, où Chroma a enregistré les extraits et leurs vecteurs.

La ligne suivante crée un `Rag` et lui donne ce nom de dossier. Les deux `vector_db_path` ne jouent pas le même rôle. À gauche du `=`, c’est le nom du paramètre attendu par `__init__`. À droite, c’est la variable définie juste au-dessus, donc le texte `"my_vector_db"`.

Ce texte arrive dans le constructeur, puis est passé tel quel à `VectorDB`. `VectorDB` ouvre le dossier `my_vector_db` s’il existe déjà sur le disque. `Rag` ne passe pas de corpus : au moment d’une question, ce dossier doit déjà avoir été construit.

## `super().__init__()` crée `self.client`

`super().__init__()` appelle le constructeur de la classe parente, `Agent`.

`Rag` hérite d'`Agent` (`class Rag(Agent)`). Comme `Rag` définit son propre `__init__`, Python n'exécute pas celui d'`Agent` tout seul. La ligne `super().__init__()` le déclenche. Dans `agent.py`, ce constructeur fait une seule chose : créer `self.client`, le client Groq, avec la clé d'API.

Ensuite, la ligne suivante ajoute ce qui est propre à `Rag` : `self.vector_db_object`, la base vectorielle ouverte avec le chemin reçu.

Après ces deux lignes, l'objet a donc les deux pièces dont `ask_rag` a besoin. `self.client` vient d'`Agent`, et `self.vector_db_object` vient de `Rag`. Sans `super().__init__()`, `self.client` n'existerait pas, et l'appel au modèle planterait.

Il n'y a pas de `self.agent`. `Agent` est le nom de la classe parente. `Rag` en hérite, donc l'objet construit avec `Rag(...)` est lui-même un `Agent`. Il ne contient pas un autre objet rangé dans `self.agent`.

`super().__init__()` exécute le constructeur d'`Agent` sur cet objet. Ce constructeur écrit `self.client = Groq(api_key=GROQ_API_KEY)`. `self.client` est le client de l'API Groq, posé directement sur le `Rag`. C'est lui que `ask_rag` utilise avec `self.client.chat.completions.create(...)`.
