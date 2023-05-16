from langchain.chains import LLMChain
from langchain import PromptTemplate
from utils import utils


def deal_noc(dialog):
    template = """请参考以下的 故障总结模板 来记录故障情况,并勿添加其他内容：

	故障发生时间：
	故障发现时间：
	故障恢复时间：
	故障持续时间：
	影响情况：一个或两个主要影响
	故障原因：一个或两个主要原因
	故障原因分类：[软件故障、人为失误、违规操作、变更操作、配置失误、服务器故障、网络故障、IDC故障、容量问题、外部依赖、第三方组件、磁盘容量不足、压力测试、CPU故障、内存故障、磁盘IO故障、其他]（请选择其中一项）
	临时解决方案：

	{dialog}
	"""
    prompt = PromptTemplate(input_variables=["dialog"], template=template)
    return LLMChain(llm=utils.Model, prompt=prompt.format(dialog=dialog))
