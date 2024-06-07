from ragable.agent import get_openai_agent
from ragable.runnable import Runnable
from ragable.adapters.qdrant import QdrantAdapter
from ragable.embedders import StandardEmbedder
import os
import logging


if __name__ == "__main__":
    rag_store = QdrantAdapter("documents")

    # Injest Document data to supply as a RAG context.
    doc_embedder = StandardEmbedder(rag_store, loglevel=logging.INFO)
    doc_embedder.train_from_text("/home/kevin/Documents/personal/Robotics and coding 2023.docx")
    # doc_embedder.train_from_text(rag_store, "I am a super saiyan")

    # agent = get_openai_agent(verbose=True)

    # tasks = [
    #     Runnable(
    #         Name="Drinks menu",
    #         Instruction="When the user asks about our drinks menu",
    #         Func=lambda params: "We serve a wide variety of soft drinks including: Coke, Coke Zero, Fanta, Appletizer",
    #         AskLLM=True
    #     ),
    #     Runnable(
    #         Name="Golange related",
    #         Instruction="When the user asks about information relating to golang.",
    #         Func=rag_store
    #     )
    # ]

    # agent.add_tasks(tasks)

    # agent.add_message("You are a helpful restaurant assistant", "system")

    # question = "Suggest a drink for a diabetic?"
    # response = agent.invoke(question)