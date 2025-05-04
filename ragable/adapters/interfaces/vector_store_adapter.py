from abc import abstractmethod
from typing import Protocol


class VectorStoreAdapter(Protocol):
    @abstractmethod
    def __init__(self, namespace, dsn=None, embedder=None):
        raise NotImplementedError

    @abstractmethod
    def add_document(self, idx, text, metadata):
        raise NotImplementedError

    @abstractmethod
    def find_documents(self, text, limit=10, filters=None):
        raise NotImplementedError

    @abstractmethod
    def get_context_data(self, text, limit=10, filters=None) -> str:
        raise NotImplementedError
