from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .openai import OpenAIAdapter
from interfaces.vector_store_adapter import VectorStoreAdapter

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

    def add_document(self, idx, text, metadata):
        vector = self.embedder.get_embeddings(text)

        metadata["raw_text"] = text
        self.store.upload_points(
            collection_name=self.namespace,
            points=[
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=metadata
                )
            ],
        )

    def find_documents(self, text, limit=10, filters=None):
        vector = self.embedder.get_embeddings(text)
        results = []
        kwargs = {
            "collection_name" : self.namespace,
            "query_vector": vector,
            "limit": limit
        }

        if filters is not None:
            kwargs['query_filter'] = filters

        found = self.store.search(**kwargs)
        results += found

        return results
