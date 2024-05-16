from typing import Dict, TypedDict,Optional, Callable

class Runnable(TypedDict):
    instruction: str
    name: str
    func: Callable
    params: Optional[Dict] = {}
    ask_llm: Optional[bool] = False

class IntentDeterminer():
    def get_intent_prompt(self, func_descritions, intents):
        return f"""
        Given the following intents, use their descriptions to determine which intent best matches the given user's message.
        <context>
        {func_descritions}
        </context>
        Please ensure your response is only one of the following and no extra text:
        {intents}
        """

    def get_intent(self, model, question, runnables : Runnable):
        func_descritions = ""
        intents = ""
        intentMappings = {}

        for runnable in runnables:
            func_descritions += f"""**{runnable['name']}**: {runnable['instruction']}"""
            intents += f"""- {runnable['name']}"""
            intentMappings[runnable['name']] = runnable

        prompt = self.get_intent_prompt(func_descritions, intents)
        print(prompt)

        response = model.invoke([
            ("system", prompt),
            ("user", question),
        ])

        intent = response.content.replace("-", "").strip()
        if intent in intentMappings.keys():
            return intentMappings[intent]

        return None