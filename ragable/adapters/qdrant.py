from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from ragable.adapters.openai import OpenAIAdapter
from ragable.adapters.interfaces.vector_store_adapter import VectorStoreAdapter
from uuid import uuid4

class QdrantAdapter(VectorStoreAdapter):
    store = None
    embedder = None
    namespace = None
    dsn = None

    def __init__(self, namespace, dsn=None, embedder=None):
        self.namespace = namespace
        self.embedder = embedder if embedder is not None else OpenAIAdapter()
        self.dsn = dsn if dsn is not None else "http://127.0.0.1:6333"

        self.store = QdrantClient(self.dsn)
        if self.store.collection_exists(self.namespace) == False:
            self.store.create_collection(
                collection_name=self.namespace,
                vectors_config=VectorParams(size=self.embedder.get_embedding_dimensions(), distance=Distance.COSINE),
            )

    def add_document(self, text, idx=None, metadata = None):
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
                    payload=metadata if metadata is not None else {}
                )
            ],
        )

    def find_documents(self, text, limit=10):
        vector = self.embedder.get_embeddings(text)
        results = []
        kwargs = {
            "collection_name" : self.namespace,
            "query_vector": vector,
            "limit": limit
        }

        found = self.store.search(**kwargs)
        results += found

        return results

    def get_context_data(self, text, limit=10):
        results = self.find_documents(text, limit)
        context = ""

        if results:
            for doc in results:
                context += "\n" + doc.payload['raw_text']

        return context