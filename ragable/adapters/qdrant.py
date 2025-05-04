import logging
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ragable.adapters.interfaces.vector_store_adapter import VectorStoreAdapter
from ragable.adapters.openai import OpenAIAdapter


class QdrantAdapter(VectorStoreAdapter):
    def __init__(
        self, namespace, dsn=None, embedder=None, api_key=None, loglevel=logging.ERROR
    ):
        self.namespace = namespace
        self.embedder = embedder if embedder is not None else OpenAIAdapter()
        self.dsn = dsn if dsn is not None else "http://127.0.0.1:6333"

        logging.basicConfig(level=loglevel, handlers=[logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        self.store = QdrantClient(self.dsn, api_key=api_key)

        if self.store.collection_exists(self.namespace) == False:
            self.store.create_collection(
                collection_name=self.namespace,
                vectors_config=VectorParams(
                    size=self.embedder.get_embedding_dimensions(),
                    distance=Distance.COSINE,
                ),
            )

    def add_document(self, text, idx=None, metadata=None):
        try:
            vector = self.embedder.get_embeddings(text)
            if metadata is None:
                metadata = {}

            metadata["raw_text"] = text
            self.store.upload_points(
                collection_name=self.namespace,
                points=[
                    PointStruct(
                        id=str(uuid4()) if idx is None else idx,
                        vector=vector,
                        payload=metadata if metadata is not None else {},
                    )
                ],
            )
        except Exception as ex:
            self.logger.error(
                "[Qdrant Adapter] Failed to store text embedding with error:", ex
            )

    def find_documents(self, text, limit=20):
        results = []
        try:
            vector = self.embedder.get_embeddings(text)
            kwargs = {
                "collection_name": self.namespace,
                "query_vector": vector,
                "limit": limit,
            }

            found = self.store.search(**kwargs)
            results += found
        except Exception as ex:
            self.logger.error(
                "[Qdrant Adapter] Failed to search for documents with error:", ex
            )

        return results

    def get_context_data(self, text, limit=20):
        results = self.find_documents(text, limit)
        context = ""

        if results:
            for doc in results:
                context += "\n" + doc.payload["raw_text"]

        return context
