from agents import OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI

azure_client = AsyncAzureOpenAI()
azure_model = OpenAIChatCompletionsModel(model="gpt-4o-mini-2024-07-18", openai_client=azure_client)
