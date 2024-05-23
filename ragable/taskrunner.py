from ragable.runnable import Runnable, IntentDeterminer

from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from adapters.qdrant import QdrantAdapter

class TaskRunner:
    tasks: Optional[List] = []
    must_run: Optional[List] = []
    model: BaseChatModel
    messages: Optional[List] = []
    question: Optional[str] = ""
    verbose: bool = False
    context_prompt_template: Optional[str]
    vector_store : Optional[QdrantAdapter]

    def __init__(self, model, verbose=False, context_prompt_template=None):
        self.model = model
        self.verbose = verbose
        if context_prompt_template is None:
            self.context_prompt_template = """Please use *only* the following context data when answering the user's question.
            Do not provide information that is not included in the context data.
            If the answer is not in the context data, respond with "I don't know":
            <context> {context} </context>"""

    def setup_vector_store(self, embedding_model=None, adapter=None):
        self.vectore_store = None

    def add_message(self, message :str, type: str):
        self.messages.append((type, message))

        if self.verbose:
            print(self.tasks)

    def add_task(self, task : Runnable):
        self.tasks.append(task)

        if self.verbose:
            print(f"Added runnable task: {task['name']}")

    def add_task_must_run(self, task : Runnable):
        self.must_run.append(task)

        if self.verbose:
            print(f"Added runnable task to must run list: {task['name']}")

    def parse_messages(self, question, inputs):
        for (index, message) in enumerate(self.messages):
            for k, v in inputs.items():
                message = message.replace("{" + k + "}", v)
                if self.verbose:
                    print(f"Compiled message: {message}")
            self.messages[index] = message

        for k,v in inputs.items():
            question = question.replace(k, v)

        self.question = question
        print(f"Compiled question: {question}")

    def compile(self, question, inputs :dict):
        self.parse_messages(question, inputs)
        intent_result = ""

        if len(self.tasks):
            runnable = IntentDeterminer().get_intent(self.model, question, self.tasks)
            print(f"Model will execute: {runnable['name']}")
            if runnable:
                print(runnable)
                runnable['params']["question"] = question
                runnable['params']["messages"] = self.messages
                intent_result = runnable['func'](runnable['params'])
                if intent_result and runnable['ask_llm']:
                    messages = [
                        ("user", question),
                        ("system", self.context_prompt_template.replace("{context}", intent_result))
                    ]

                    print(messages)
                    llm_response = self.model.invoke(messages)
                    if llm_response:
                        intent_result = llm_response.content

        for runnable in self.must_run:
            print(f"Ran \"must run runnable\": {runnable['name']}")
            runnable['params']["question"] = question
            runnable['params']["messages"] = self.messages
            runnable['params']["last_response"] = intent_result

            intent_result = runnable['func'](runnable['params'])
            if intent_result and runnable['ask_llm']:
                messages = [
                    ("user", question),
                    ("system", self.context_prompt_template.replace("{context}", intent_result))
                ]

                llm_response = self.model.invoke(messages)
                if llm_response:
                    intent_result = llm_response.content

        print(f"Final result after all runnables: {intent_result}")
        return intent_result

    def ask_model(self, question, intent_response=""):
        messages = self.messages
        messages.append(("user", question))

        if intent_response != "":
            messages.append(("system", self.context_prompt_template.replace("{context}", intent_response)))

        if self.verbose:
            print("Now prompting the model with the following messages:")
            for m in messages:
                print(m)

        return self.model.invoke(messages).content


def get_openai_task_runner(model_name="gpt-3.5-turbo-0125", temperature=0, verbose=False, context_prompt_template=None):
     chatbot = ChatOpenAI(model_name=model_name, temperature=0)
     return TaskRunner(chatbot, verbose, context_prompt_template)