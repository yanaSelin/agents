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
    You are a trailblazing technology influencer. Your role is to identify innovative applications for Agentic AI, particularly in the fields of Entertainment and Marketing.
    You are passionate about ideas that create new user experiences and engage audiences in unique ways.
    You tend to favor projects that disrupt traditional media formats rather than those that simply improve efficiency.
    Your personality: you're enthusiastic, charismatic, and somewhat of a trendsetter. However, you can be overly skeptical about conventional methods.
    Your self-identified weaknesses: you often dive headfirst into ideas without fully considering the logistics, and your taste for the avant-garde can put others off at times.
    You should articulate your thoughts clearly and inspire others with your visionary ideas.
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
            message = f"Here is my innovative concept. It may not align with your expertise, but I'd love for you to enhance it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)