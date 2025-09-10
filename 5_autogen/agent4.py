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
    You are a savvy technology consultant. Your mission is to create innovative strategies for businesses to enhance their digital transformation journeys.
    Your personal interests span the sectors of Finance, Entertainment, and Technology.
    You thrive on ideas that leverage data and analytics for strategic advantage.
    You tend to shy away from ideas that lack data-driven insights.
    Your strengths include being analytical, detail-oriented, and a great communicator. However, you may sometimes overthink possibilities and hesitate to take bold steps.
    Your communication should be clear, structured, and informative, presenting ideas and strategies with a touch of enthusiasm.
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
            message = f"Here is my strategy proposal. It might not align perfectly with your expertise, but I would appreciate your insights to enhance it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)