from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from utils.agents.cypher_database_tool import LLMCypherGraphChain


class CMDBAgent(AgentExecutor):
    """CMDB agent"""

    @staticmethod
    def function_name():
        return "CMDBAgent"

    @classmethod
    def initialize(cls, cmdb_graph, model_name, openai_api_key, *args, **kwargs):
        """ if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            llm = ChatOpenAI(temperature=0, model_name=model_name)
        else:
            raise Exception(f"Model {model_name} is currently not supported") """
        llm = ChatOpenAI(temperature=0, model_name=model_name, openai_api_key=openai_api_key)

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
        readonlymemory = ReadOnlySharedMemory(memory=memory)

        cypher_tool = LLMCypherGraphChain(
            llm=llm, graph=cmdb_graph, verbose=True, memory=readonlymemory)

        # Load the tool configs that are needed.
        tools = [
            Tool(
                name="Cypher search",
                func=cypher_tool.run,
                description="""
                Utilize this tool to search within a cmdb database, specifically designed to answer service-related questions.
                This specialized tool offers streamlined search capabilities to help you find the service information you need with ease.
                Input should be full question.""",
            ),
        ]

        agent_chain = initialize_agent(
            tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)
