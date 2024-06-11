from ragable.agent import get_openai_agent
from ragable.runnable import Runnable, runnable_from_func
from ragable.adapters.qdrant import QdrantAdapter
from ragable.embedders import StandardEmbedder

"""
Runnables are Supercharged functions that can interact with LLMs.

You can create a Runnable using the below syntax as a decorator or
you can instantiate Runnable on it's own and pass the name of your function
via the Func argument. (See bulbasaur example lower down.)

Fields available:

Name: A unique name to identify this runnable.
Instruction: A prompt that the Agent will use to determine which Runnable to execute.
AskLLM: True by default. Will send the function's output to the LLM for further reasoning.
        Set to False, to return the raw function's output.
Params: Any data you want to make available to your function, it's not sent to the LLM and should
        be used internally within the function, for example you can pass a user_id or session information.
"""
@runnable_from_func(
    Name="All about php strings",
    Instruction="When the human asks about php"
)
def php_strings(params):
    response = """
        str_replace('x', 'y', $z)
        stripos($the_big_blob_of_text, $the_thing_to_search_for)
    """
    return response

@runnable_from_func(
    Name="All about legendary pokemon",
    Instruction="When the human asks about legendary pokemon"
)
def legendary_pokemon(params):
    context_data = ""
    with open("./testdata/legendary_pokemon.txt", "r") as f:
        txt = f.read()
    return context_data


if __name__ == "__main__":
    # Sets up an OpenAI powered agent.
    # Agents can register multiple tasks and will intelligently route the LLM
    # - to tasks based on the Runnable "Instruction" prompt.

    agent = get_openai_agent()

    # Easy integration with the Qdrant vector store (you will need Qdrant running locally)
    # Pass in "dsn" and "api_key" for any other setup.

    qdrant = QdrantAdapter("ragable_documents")

    # The embedder Allows you to feed most common document types into the RAG system.
    # Each document is chunked into LLM friendly chunks and vector embedded.
    embedder = StandardEmbedder(qdrant)

    # Path to your document. Optionally, you can also pass in a "doc_id".
    # The doc_id can be an integer or uuid.
    # Formats supported: txt, pdf, docx, odt, pptx, odp
    embedder.train_from_document("./testdata/bulbasaur.txt")

    # You can also embed and index regular strings.
    # doc_id is required.
    # embedder.train_from_text("some text", 1234)

    # A none decorator verson of a Runnable.
    bulbasaur_knowledge = Runnable(
        Name="Information about bulbasaur",
        Instruction="When the human asks about bulbasaur",
        Func=qdrant
    )

    # Tell the agent which Runnable functions it's allowed to execute.
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

    # Here you can feed the Agent any additional prompts as needed.
    # For example, you can store the chat history in Redis or a local session and
    # - then add each of the historical messages using this function.
    # Supported message types: system, user, ai, assistant
    agent.add_message("You are a useful informational bot.", "system")

    for q in questions:
        response = agent.invoke(q)
        print(response)