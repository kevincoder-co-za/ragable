import time

from openai import OpenAI

from ragable.adapters.interfaces.llm_adapter import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        model="gpt-3.5-turbo",
        embedding_model="text-embedding-3-small",
        temperature=0,
    ):
        self.client = OpenAI()
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.last_response = None

    def get_embedding_dimensions(self):
        if self.embedding_model in ("text-embedding-3-small", "text-embedding-ada-002"):
            return 1536
        elif self.embedding_model == "text-embedding-3-large":
            return 3072

    def get_embeddings(self, sentence):
        retries = 0
        while retries <= 3:
            try:
                response = self.client.embeddings.create(
                    input=sentence, model=self.embedding_model
                )

                return response.data[0].embedding
            except Exception as ex:
                time.sleep(2)
                retries += 1

        raise Exception("Failed to generate vector embedding.")

    def parse_agent_messages(self, messages):
        formatted_messages = []
        for m in messages:
            formatted_messages.append({"role": m[0], "content": m[1]})
        return formatted_messages

    def get_last_llm_response(self):
        return self.last_response

    def invoke(self, messages, parse_agent_messages=True):
        if parse_agent_messages:
            messages = self.parse_agent_messages(messages)
        completion = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=messages
        )

        self.last_response = completion
        return completion.choices[0].message.content
