from typing import Protocol
from abc import abstractmethod

class LLMAdapter(Protocol):
    @abstractmethod
    def __init__(self, model, embedding_model, temperature):
        raise NotImplementedError

    @abstractmethod
    def get_embedding_dimensions(self):
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, sentence):
        raise NotImplementedError

    @abstractmethod
    def parse_agent_messages(self, messages):
        raise NotImplementedError

    @abstractmethod
    def invoke(self, messages) -> str:
        raise NotImplementedError