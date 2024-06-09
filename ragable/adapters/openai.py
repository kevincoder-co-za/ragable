from openai import OpenAI
import time

class OpenAIAdapter:
    def __init__(self, model = "text-embedding-3-small"):
        self.client = OpenAI(model)
        self.model = model

    def get_embedding_dimensions(self):
        if self.model in ("text-embedding-3-small", "text-embedding-ada-002"):
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072

    def get_embeddings(self, sentence):
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
