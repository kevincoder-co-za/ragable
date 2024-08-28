# Welcome to Ragable!
Ragable is an AI library that helps you build multi-turn Chatbots with ease, the library provides several easy-to-use classes to help you build "agent style" workflows. The Agent will first analyze the user's input and then determine what function to invoke (known as Runnable's).

Each **Runnable** takes the following arguments:
 - **Name**: a unique identifier that describes this runnable.
 - **Instruction**: Instruct the LLM on when it should invoke this runnable.
 - **Func**: Any Python function you want to run if the model chooses to execute this Runnable. The function should return text as its output. You can also pass a vector store to this function, when doing so, the Runnable will query the vector store for similar documents and pass it on to the LLM for further reasoning.
 - **AskLLM**: By default the LLM will simply execute the function and return its output to the user. If you prefer to pass on the function output to the LLM for further reasoning, set this to **True**.
 - **Params**: Any local data you want to make available to the Runnable function. This can be used to pass a user object or session information. Params are not sent to the LLM.

```python
from ragable.agent import get_openai_agent
from ragable.runnable import Runnable
from ragable.adapters.qdrant import QdrantAdapter
from ragable.embedders import StandardEmbedder
import os

# See examples.py for more details
    agent = get_openai_agent()
    qdrant = QdrantAdapter("ragable_documents")
    embedder = StandardEmbedder(qdrant)
    embedder.train_from_document("./testdata/bulbasaur.txt")

    bulbasaur_knowledge = Runnable(
        Name="Information about bulbasaur",
        Instruction="When the human asks about bulbasaur",
        Func=qdrant
    )

    agent.add_tasks([
        legendary_pokemon,
        php_strings,
        bulbasaur_knowledge
    ])

    questions = [
        "What is a legendary pokemon?",
        "How to perform a string replace in PHP?",
        "How to find a string in another string in PHP?",
        "Which Pokemon are the evolved forms of bulbasaur?"
    ]

    for q in questions:
        response = agent.invoke(q)
        print(response)
```
## How to install?

You can GIT clone this repository and run:

    pip install requirements.txt

Thereafter, you can then import any of the Ragable classes into your Python projects.

**Note:** If you plan on using a vector store with Ragable, we currently support Qdrant. You can use their docker image to setup Qdrant as follows:
```python
docker run -dit --name raggable-qdrant -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant
```

## Bulk import and vectorize documents

To bulk import documents into Qdrant, simply run the following with the relevant folder path and collection name:
```python
 python document_feeder.py --folder ./documents/  --collection pokemon
```
