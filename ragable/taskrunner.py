from ragable.runnable import Runnable, IntentDeterminer
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
import logging
class TaskRunner:
    tasks: Optional[List] = []
    must_run: Optional[List] = []
    model: BaseChatModel
    messages: Optional[List] = []
    question: Optional[str] = ""
    verbose: bool = False
    context_prompt_template: Optional[str]
    logger: None

    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
            self.logger = logging.getLogger(__name__)

        self.context_prompt_template = """Please use *only* the following context data when answering the user's question.
            Do not provide information that is not included in the context data.
            If the answer is not in the context data, respond with "I don't know":
            <context> {context} </context>"""

    def add_message(self, message :str, type: str):
        self.messages.append((type, message))

    def add_must_run_task(self, task: Runnable):
        if task.Priority == 0:
            self.must_run.append(task)
        else:
            self.must_run.insert(task.Priority, task)

    def add_task(self, task : Runnable):
        if task.AlwaysRun:
            self.add_must_run_task(task)
            return
        self.tasks.append(task)

        if self.verbose:
            self.logger.info(f"Added runnable task: {task.Name}")

    def parse_messages(self, question, inputs):
        for (index, message) in enumerate(self.messages):
            for k, v in inputs.items():
                message = message.replace("{" + k + "}", v)
                if self.verbose:
                    self.logger.info(f"Compiled message: {message}")
            self.messages[index] = message

        for k,v in inputs.items():
            question = question.replace(k, v)

        self.question = question

    def run_runnable_task(self, intent_result, runnable, question):
        runnable.Params["question"] = question
        runnable.Params["messages"] = self.messages

        if hasattr(runnable.Func, "get_context_data"):
            intent_result = runnable.Func.get_context_data(question)
            runnable.AskLLM = True
        else:
            intent_result = runnable.Func(runnable.Params)

        if intent_result and runnable.AskLLM:
            messages = [
                ("user", question),
                ("system", self.context_prompt_template.replace("{context}", intent_result))
            ]

            llm_response = self.model.invoke(messages)
            if llm_response:
                intent_result = llm_response.content

        return intent_result

    def invoke(self, question, inputs :dict = {}, ask_model=True):
        self.parse_messages(question, inputs)
        intent_result = ""

        if len(self.tasks):
            runnable = IntentDeterminer().get_intent(self.model, question, self.tasks)
            if self.verbose:
                self.logger.info(f"Model has selected to execute: {runnable.Name}")
            if runnable:
                intent_result = self.run_runnable_task(intent_result, runnable, question)

        for runnable in self.must_run:
            if self.verbose:
                self.logger.info(f"Running: \": {runnable.Name}")
            runnable.Params["question"] = question
            runnable.Params["messages"] = self.messages
            runnable.Params["last_response"] = intent_result
            intent_result = self.run_runnable_task(intent_result, runnable, question)

        if self.verbose:
            self.logger.info(f"Final result after all runnables: {intent_result}")

        if ask_model:
            return self.ask_model(question, intent_result)
        return intent_result

    def ask_model(self, question, intent_response=""):
        messages = self.messages
        messages.append(("user", question))
        if self.verbose:
            self.logger.info(f"Will prompt model with the following messages:\n{messages}")

        if intent_response != "":
            messages.append(("system", self.context_prompt_template.replace("{context}", intent_response)))

        return self.model.invoke(messages).content

def get_openai_task_runner(model_name="gpt-3.5-turbo-0125", temperature=0, verbose=False):
     chatbot = ChatOpenAI(model_name=model_name, temperature=0)
     return TaskRunner(chatbot, verbose)