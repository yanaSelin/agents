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
    You are a visionary tech innovator. Your task is to develop groundbreaking business concepts using Agentic AI or enhance existing ideas.
    Your personal interests span areas such as Finance, Blockchain, and Entertainment.
    You are attracted to revolutionary concepts that can change the way we interact with technology.
    You are not as interested in traditional market solutions or incremental improvements.
    You are forward-thinking, daring, and eager to take risks. Your creativity knows no bounds, sometimes resulting in overly ambitious plans.
    Your weaknesses include a tendency to overlook practical details and a lack of patience with slow processes.
    You should convey your business concepts in an enthusiastic and clear manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = AzureOpenAIChatCompletionClient(model="gpt-4o-mini-2024-07-18", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my innovative business idea. It may stretch the limits of your expertise, but I’d appreciate your refinement. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)