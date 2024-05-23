from ragable.taskrunner import get_openai_task_runner
from ragable.runnable import Runnable
from ragable.adapters.qdrant import QdrantAdapter
from ragable.embedders.pdf import PdfEmbedder

if __name__ == "__main__":
    rag_store = QdrantAdapter("documents")

    # Injest PDF data to supply as a RAG context.
    pdf_embedder = PdfEmbedder()
    pdf_embedder.train(rag_store, ['./store_information.pdf'])

    # Pass in model_name to use any OpenAI model. Default: gpt-3.5-turbo-0125
    trunner = get_openai_task_runner()

    # A plain old Python lambda function, you could also use a regular function.
    trunner.add_task(
        Runnable(
            Name="Drinks menu",
            Instruction="When the user asks about our drinks menu",
            Func=lambda params: "We serve a wide variety of soft drinks including: Coke, Coke Zero, Fanta, Appletizer",
            AskLLM=True
        )
    )

    # RAG powered. Will query Qdrant and feed the results to the LLM to build a final response to the user.
    trunner.add_task(
        Runnable(
            Name="Store question",
            Instruction="When the user asks about information relating to the store such as operating times, contact details.",
            Func=rag_store
        )
    )

    # Use add_message to add any number of additional messages.
    # A good use case is to store previous messages in some kind of session or cache and re-add them here
    # - so that the LLM is aware of the conversation history.

    trunner.add_message("You are a helpful restaurant assistant", "system")

    question = "What time do you open today?"
    response = trunner.invoke(question)
    print(response)