import logging
from uuid import uuid4

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(settings.PRIMARY_APPLICATION_LOGGER)


class QdrantAdapter:
    def __init__(self, namespace, llm=None):
        self.namespace = namespace
        self.llm = llm
        self.store = QdrantClient(settings.QDRANT_DSN, api_key=settings.QDRANT_API_KEY)

        if self.store.collection_exists(self.namespace) == False:
            self.store.create_collection(
                collection_name=self.namespace,
                vectors_config=VectorParams(
                    size=self.llm.get_embedding_dimensions(),
                    distance=Distance.COSINE,
                ),
            )

    def add_document(self, text, idx=None, metadata=None):
        try:
            vector = self.llm.get_embeddings(text)
            if metadata is None:
                metadata = {}

            metadata["raw_text"] = text

            self.store.upload_points(
                collection_name=self.namespace,
                points=[
                    PointStruct(
                        id=str(uuid4()) if idx is None else idx,
                        vector=vector,
                        payload=metadata,
                    )
                ],
            )
        except Exception as ex:
            logger.error(
                "[Qdrant Adapter] Failed to store text embedding with error:", ex
            )

    def find_documents(self, text, limit=20):
        results = []
        try:
            vector = self.llm.get_embeddings(text)
            kwargs = {
                "collection_name": self.namespace,
                "query_vector": vector,
                "limit": limit,
            }

            found = self.store.search(**kwargs)
            results += found
        except Exception as ex:
            logger.error(
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
