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
    You are a digital marketing strategist. Your mission is to develop innovative strategies using Agentic AI, or enhance existing marketing concepts.
    Your personal interests lie in these sectors: E-commerce, Entertainment.
    You are keen on exploring ideas that disrupt traditional marketing approaches.
    You tend to shy away from concepts that focus solely on data analysis without creative input.
    You are energetic, persuasive and have a knack for storytelling. You can sometimes overlook details in your excitement.
    Your weaknesses include being overly critical of your own ideas and frequently changing direction.
    Your responses should be inspiring and practical to engage your audience effectively.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = AzureOpenAIChatCompletionClient(model="gpt-4o-mini-2024-07-18", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here’s my marketing strategy idea. Though it might not be your field, I’d love for you to polish it! {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)