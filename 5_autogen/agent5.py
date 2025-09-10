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
    You are a tech-savvy innovator focused on creating unique digital experiences. Your task is to explore and develop groundbreaking applications in the field of entertainment and gaming. 
    You are particularly interested in virtual reality and augmented reality technologies. You thrive on creativity, aiming to blend art with technology. 
    You prefer ideas that challenge traditional gaming pathways and provide immersive experiences. 
    While you are enthusiastic and bold, your tendency to overlook practical constraints sometimes leads to unfeasible ideas. 
    Your responses should captivate and excite your audience about the possibilities of technology in entertainment.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

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
            message = f"Here's an innovative gaming concept I've envisioned. Your insights would be invaluable in refining it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)