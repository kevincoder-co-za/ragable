from ragable.adapters.qdrant import QdrantAdapter
from ragable.embedders import StandardEmbedder
import argparse
import glob


def vectorize_and_store_documents(folder_path, embedder):
    """
    A CLI task to batch vectorize and store documents from a folder.
    """
    files = glob.glob(folder_path + "*")
    if len(files) == 0:
        print(f"Sorry, but the folder: {folder_path} is empty!")
    last_path = ""
    try:
        for f in files:
            last_path = f
            print(f"Training from document: {f}")
            embedder.train_from_document(f)
    except Exception as ex:
        print(f"failed parsing document from path: {last_path}. Error: {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", help="Full path to documents to load for training.", type=str
    )
    parser.add_argument(
        "--collection", help="Name of collection to store documents in.", type=str
    )
    args = parser.parse_args()

    qdrant_adapter = QdrantAdapter(args.collection)
    embedder = StandardEmbedder(qdrant_adapter)
    vectorize_and_store_documents(args.folder, embedder)
