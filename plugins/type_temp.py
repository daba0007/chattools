from langchain.chains import LLMChain
from langchain import PromptTemplate
from utils import utils


def type_select(question):
    template = """你的工作是对问题进行分类,只能在以下提供的分类中判断出问题的种类:
	Local: 咨询平台，地址，网址或个人信息
	Noc: 对对话进行总结
	Fess: 除了 Local 和 Noc以外的种类

	示例:
	问题: 百度的地址是多少
	回答: Local

	问题: 介绍下cschat
	回答: Fess

	% QUESTION
	问题: {question}
	回答: 
	"""
    prompt = PromptTemplate(input_variables=["question"], template=template)
    return LLMChain(llm=utils.Model, prompt=prompt.format(question=question))
