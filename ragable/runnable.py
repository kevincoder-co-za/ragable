from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class Runnable:
    Instruction: str
    Name: str
    Func: Callable
    Params: Dict = field(default_factory=dict)
    AskLLM: Optional[bool] = True


def runnable_from_func(**kwargs):
    def wrapper(func):
        return Runnable(Func=func, **kwargs)

    return wrapper


class IntentDeterminer:
    def get_intent_prompt(self, func_descritions, intents):
        return f"""
        Given the following intents, use their descriptions to determine which intent best matches the given user's message.
        <context>
        {func_descritions}
        </context>
        Please ensure your response is only one of the following and no extra text:
        {intents}
        """

    def get_intent(self, model, question, runnables: Runnable):
        func_descritions = ""
        intents = ""
        intentMappings = {}

        for runnable in runnables:
            func_descritions += f"""**{runnable.Name}**: {runnable.Instruction}"""
            intents += f"""- {runnable.Name}"""
            intentMappings[runnable.Name] = runnable

        prompt = self.get_intent_prompt(func_descritions, intents)
        response = model.invoke(
            [
                ("system", prompt),
                ("user", question),
            ]
        )

        intent = response.replace("-", "").strip()
        if intent in intentMappings.keys():
            return intentMappings[intent]

        return None
