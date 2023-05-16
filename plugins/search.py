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

# Bing查询工具 ，用于搜索Bing
class FessTools(BaseTool):
    name = "搜索 Fess"
    description = "搜索除了咨询网站地址和个人信息以外的信息"

    def _run(self, query: str) -> str:
        return fess_search.find(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

# Fess查询工具 ，用于搜索Fess
class FessTools(BaseTool):
    name = "搜索 Fess"
    description = "搜索除了咨询网站地址和个人信息以外的信息"

    def _run(self, query: str) -> str:
        return fess_search.find(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

class LocalTools(BaseTool):
    name = "搜索 Local"
    description = "搜索网站地址和个人信息"

    def _run(self, query: str) -> str:
        return local_search.find(query)

from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
 
template="""在回答问题之前，必须使用以下格式分析解决问题并适当使用工具:
{tools}

问题: 你必须回答的问题
思考: 对于问题以及已有的信息进行的思考
操作: 要采取的动作应该是[{tool_names}]的其中一种
输入: 操作的结果
观察: 查看输入的内容是有相关信息,如果有可以提炼出来得出最终答案
...(此 思考/操作/输入/观察 可以重复多次直到得出答案)
最终回答: 原始输入问题的最终答案

示例:
问题: 如何访问百度
思考: 我不知道百度的地址
操作: Local
输入: 百度的地址是多少
观察: 百度: https://www.baidu.com/
最终回答: https://www.baidu.com/

开始!

问题: {input}
{agent_scratchpad}"""

""" tools=[Tool(
        name = "Fess",
        func=fess_search.find,
        description="搜索除了咨询网站地址和个人信息以外的信息"
    ),Tool(
        name = "Loca",
        func=local_search.find,
        description="搜索网址,地址,平台以及个人信息"
    )] """

def test(s):
    return "不知道"

tools=[Tool(
        name = "Local",
        func=local_search.find_with_str,
        description="可以在本地进行搜索，咨询平台，地址，网址或个人信息时使用的工具"
    ),Tool(
        name = "Bing",
        func=test,
        description="联网搜索使用的工具,遇到不知道的事情，无法预测和推测的事情时就要用这个工具搜索"
    )]

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
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
        return self.template.format(**kwargs)
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "最终回答:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("最终回答:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"操作\s*\d*\s*:(.*?)\n输入\s*\d*\s*:[\s]*(.*)"
        print(utils.Green, llm_output, utils.White)
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
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