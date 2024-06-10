from ragable.adapters.openai import OpenAIAdapter
import os
import logging


if __name__ == "__main__":
    model = OpenAIAdapter()
    print(model.invoke([("user", "What is the weather today in Durban?")]))