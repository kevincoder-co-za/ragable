from openai import OpenAI
import time

class OpenAIAdapter:
    client = None
    def build_openai_client(self):
        if not self.client:
            self.client = OpenAI()

    def __init__(self, model = "text-embedding-3-small"):
        self.build_openai_client()
        self.model = model

    def get_embedding_dimensions(self):
        return 1536

    def get_embeddings(self, sentence):
        self.build_openai_client()

        retries = 0
        while retries <= 3:
            try:
                response = self.client.embeddings.create(
                    input=sentence,
                    model=self.model
                )

                return response.data[0].embedding
            except Exception as ex:
                time.sleep(2)
                retries += 1

        raise Exception("Failed to generate vector embedding.")
