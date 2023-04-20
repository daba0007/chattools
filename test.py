""" from langchain.document_loaders import GoogleDriveLoader
    loader = GoogleDriveLoader(
        folder_id="1qnyg0NObislAYODBU2CuzJX4-07QCGHi8350XLRJM9U",
        recursive=False
    ) 
    from langchain import SQLDatabase, SQLDatabaseChain
    db = SQLDatabase.from_uri(utils.Clickhouse)
    db_chain = SQLDatabaseChain(llm=utils.Model, database=db, verbose=True)
    print(db_chain.run("数据库中有多少个表"))
    # docs = loader.load()
    from langchain.document_loaders import TextLoader
    loader = TextLoader('state_of_the_union.txt', encoding='utf8')
    docs = loader.load()
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    if len(texts) > 0 :
        from langchain.vectorstores import Chroma
        db = Chroma.from_documents(texts)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=utils.Model, chain_type="stuff", retriever=retriever)
        query = "简单说下 inhouse 双活方案的问题点有哪些"
        print(qa.run(query))
    else:
        print("数据为空") """
        
        
""" # google
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
loader = GoogleDriveLoader(
    document_ids=["1qnyg0NObislAYODBU2CuzJX4-07QCGHi8350XLRJM9U"],
    credentials_path="credentials.json",
    token_path="token.json",
    recursive=False
)
docs = loader.load()
print(docs)
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
if len(texts) > 0 :
    from langchain.vectorstores import Chroma
    db = Chroma.from_documents(texts)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(llm=utils.Model, chain_type="stuff", retriever=retriever)
    query = "简单说下 inhouse 双活方案的问题点有哪些"
    print(qa.run(query))
else:
    print("数据为空") """
    
""" from utils.main import setting, load_LLM
import yaml
from utils import utils

data = '2023-04-17 13:28:28 | Zhi Wenyu (支文宇) | NOC & PJM : Supervisor那个面板是20分钟聚合数据的，改短一点能看到17分恢复'
###

from kor import create_extraction_chain
from kor import from_pydantic
from pydantic import BaseModel, Field

class Issue(BaseModel):
    time: str = Field(description="时间")
    user: int = Field(description="说话人")

# 读取环境变量
with open('config.yaml', 'r', encoding='utf-8') as f:
    setting(yaml.safe_load(f))
# Load Model
load_LLM()
utils.Model.load_model()
print(utils.Green, "模型加载完成", utils.White)

schema, validator = from_pydantic(
    Issue,
    description="故障信息",
    many=True,
    examples=[("2023-04-17 12:15:46 | Wang Shuo (王烁) | CSSRE | https://i.shp.ee/vxf58rp : 发送消息数掉了", {"user": "Wang Shuo (王烁)", "time": "2023-04-17 12:15:46"}),
              ("2023-04-17 12:19:55 | Lin Zhaoyu (林兆宇) : 应该是进线无法分配了导致的", {"user": "Lin Zhaoyu (林兆宇)", "time": "2023-04-17 12:19:55"}),
              ("2023-04-17 12:24:18 | Kevin Lin | 林志衡 : 目前 Realtime Sync 还能工作。但如果任务有出错重试时，该任务就不工作了", {"user": "Kevin Lin", "time": "2023-04-17 12:24:18"})],
)
chain = create_extraction_chain(
    utils.Model, schema, encoder_or_encoder_class="csv", validator=validator
)
print("开始测试")
chain.prompt.format_prompt(text="[user input]").to_string()
print(chain.predict_and_parse(text=data)["validated_data"]) """