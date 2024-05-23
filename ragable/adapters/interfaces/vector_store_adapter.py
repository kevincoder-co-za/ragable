from typing import Protocol
from abc import abstractmethod

class VectorStoreAdapter(Protocol):
    @abstractmethod
    def  __init__(self, namespace, dsn=None, embedder=None):
        raise NotImplementedError

    @abstractmethod
    def add_document(self, idx, text, metadata):
        raise NotImplementedError

    @abstractmethod
    def find_documents(self, text, limit=10, filters=None):
        raise NotImplementedError