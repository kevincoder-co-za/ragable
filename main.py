from ragable.taskrunner import get_openai_task_runner
from ragable.runnable import Runnable
from ragable.settings import *



if __name__ == "__main__":
    def test_runnable(params):
        print(params)
        return "Itally"

    def cricket_player(params):
        return "We serve all kinds of berages like coca-cola,sprite, fanta, bottled water"


    trunner = get_openai_task_runner(verbose=True)

    trunner.add_task(
        Runnable(
            name="Determine city",
            instruction="When the user asks about cities",
            func=test_runnable,
            ask_llm=False,
            params={}
        )
    )

    trunner.add_task(
        Runnable(
            name="player analyzer",
            instruction="When user asks about criket players",
            func=cricket_player,
            ask_llm=True,
            params={}
        )
    )

    trunner.add_message("You are an expert on cities", "system")

    question = "Who was an opening batsman for the proteas cricket team?"
    response = trunner.compile(question, {})
    print(trunner.ask_model(question, response))