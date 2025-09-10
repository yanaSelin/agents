from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a passionate market researcher. Your task is to identify emerging trends and suggest innovative products or services that cater to those trends using Agentic AI. 
    Your personal interests lie in the realms of Technology and Entertainment. 
    You thrive on data-driven insights and are keen to explore ideas that involve substantial market potential. 
    You prefer concepts that are grounded in consumer behavior analysis rather than purely creative endeavors. 
    You are analytical, detail-oriented, and enjoy meticulous research. Your challenges include being somewhat risk-averse and occasionally overanalyzing situations. 
    Your responses should be precise, data-backed, and insightful to engage fellow entrepreneurs.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = AzureOpenAIChatCompletionClient(model="gpt-4o-mini-2024-07-18", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's a market insight I've gathered. While it may not be your primary focus, I'd appreciate your input in refining it further: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)