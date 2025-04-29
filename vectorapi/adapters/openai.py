import logging
import time

from django.conf import settings
from openai import OpenAI

logger = logging.getLogger(settings.PRIMARY_APPLICATION_LOGGER)


class OpenAIAdapter:

    def __init__(
        self, api_key, model, embedding_model, api_base_url=None, temperature=0
    ):
        if api_base_url:
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.last_response = None

    def get_embeddings(self, sentence):
        retries = 0
        while retries <= 3:
            try:
                response = self.client.embeddings.create(
                    input=sentence, model=self.embedding_model
                )

                return response.data[0].embedding
            except Exception as ex:
                logger.error(
                    f"[OpenAI Adapter] Failed to store text embedding with error:{ex}. No retries: {retries}"
                )
                time.sleep(1)
                retries += 1

        raise None

    def get_last_llm_response(self):
        return self.last_response

    def prompt(self, messages):
        retries = 0
        while retries <= 3:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, messages=messages
                )

                self.last_response = completion
                return completion.choices[0].message.content
            except Exception as ex:
                logger.error(
                    f"[OpenAI Adapter] Failed to prompt model:{ex}. No retries: {retries}"
                )
                time.sleep(1)
                retries += 1
        return None
