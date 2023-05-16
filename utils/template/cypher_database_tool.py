from langchain.prompts import (
    PromptTemplate
)
from utils.utils import logger

examples = """
# 谁负责 cschat
MATCH (m:Cmdb)<-[r:PIC]-(a) WHERE m.name =~ '.*cschat.*' RETURN m,r,a
# inhouse 有哪些服务
MATCH (m:Cmdb) where m.name =~ '.*inhouse.*' RETURN m
# cschat的 spex 调用了哪些服务
MATCH (o:Cmdb)<-[r:SpexServer]-(p:Spex)<-[s:SpexClient]-(q) Where o.name =~'.*cschat.*' RETURN o,r,p,s,q
# 哪些服务调用了 cschat
MATCH (o:Cmdb)<-[r:SpexServer]-(p:Spex)-[s:SpexClient]->(q) Where o.name =~'.*cschat.*' RETURN o,r,p,s,q
"""


SYSTEM_TEMPLATE = """
You are an assistant with an ability to generate Cypher queries based off example Cypher queries.
Example Cypher queries are:\n""" + examples + """\n
Do not response with any explanation or any other information except the Cypher query.
You do not ever apologize and strictly generate cypher statements based of the provided Cypher examples.
Do not provide any Cypher statements that can't be inferred from Cypher examples.
Inform the user when you can't infer the cypher statement due to the lack of context of the conversation and state what is the missing context.
question: {question}
answer:
"""

def getCypher(llm, question):
    logger.info(f"Cypher generator inputs: {question}")
    chat_prompt = PromptTemplate(input_variables=["question"],template=SYSTEM_TEMPLATE)
    cypher = llm(chat_prompt.format(question=question))
    logger.info(f"Cypher generator outputs: {cypher}")
    # If Cypher statement was not generated due to lack of context
    if not "MATCH" in cypher:
        return ''
    return cypher
