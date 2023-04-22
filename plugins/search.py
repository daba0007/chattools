from plugins.bing_search import bing_search
from plugins.multi_search import MultiSearch
from plugins.fess_search import fess_search
from plugins.local_search import local_search
from utils import utils

search_functions = {
    'mix': MultiSearch(utils.Weight).find,
    'bing': bing_search.find,
    'fess': fess_search.find,
    'local': local_search.find
}

def find(prompt, library='mix', step=1):
    return search_functions[library](prompt, step)

from langchain.tools import BaseTool

# 天气查询工具 ，无论查询什么都返回Sunny
class FessTools(BaseTool):
    name = "搜索 Fess"
    description = "搜索除了咨询网站地址和个人信息以外的信息"

    def _run(self, query: str) -> str:
        return fess_search.find(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

# 计算工具，暂且写死返回3
class LocalTools(BaseTool):
    name = "搜索 Local"
    description = "搜索网站地址和个人信息"

    def _run(self, query: str) -> str:
        return local_search.find(query)

from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
 
template="""尽可能的回答以下的问题,你有权利使用以下工具:
{tools}
使用以下的模版:
问题: 你必须回答的问题
思考: 你必须经常考虑要怎么做
操作: 你要执行的操作,必须是[{tool_names}]的其中一种
输入: 动作的输入
观察: 操作的结果
...(这个 思考/操作/输入/观察可以重复多次)

思考: 我现在想要知道最终回答
最终回答: 对于最初问题的最终的回答是

开始!

问题: {input}
{agent_scratchpad}"""

tools=[Tool(
        name = "Fess",
        func=fess_search.find,
        description="搜索除了咨询网站地址和个人信息以外的信息"
    ),Tool(
        name = "Loca",
        func=local_search.find,
        description="搜索网站地址和个人信息"
    )]

class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n观察: {observation}\n思考: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "最终回答:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("最终回答:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"操作\s*\d*\s*:(.*?)\n操作\s*\d*\s*输入\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
def setSearchAgent():
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=utils.Model, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    utils.SearchAgent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\n观察:"], 
        allowed_tools=tool_names
    )