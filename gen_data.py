from langchain.embeddings import HuggingFaceEmbeddings
import re
import os
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import sentence_transformers
from utils import utils
import yaml
import pathlib

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    utils.Gen_Data.LibraryPath = config["library"]["source_path"]
    utils.Gen_Data.Target_Path = config["library"]["source_path"] + '_out'
    source_folder_path = pathlib.Path.cwd() / config["library"]["source_path"]
    target_folder_path = pathlib.Path.cwd() / (config["library"]["source_path"] + '_out')
    utils.Gen_Data.Size = config["library"]["size"]
    utils.Gen_Data.Overlay = config["library"]["overlay"]
    utils.Gen_Data.Model_Path = config["library"]["model_path"]
    utils.Gen_Data.Device = config["library"]["device"]

if not target_folder_path.exists():
    target_folder_path.mkdir()
    
print("预处理数据")
for root, dirs, files in os.walk(source_folder_path):
    for file in files:
        file_path = pathlib.Path(root) / file
        if not file_path.is_file():
            continue
        try:
            with open(file_path, "r", encoding='utf-16') as f:
                data = f.read()
        except UnicodeError:
            with open(file_path, "r", encoding='utf-8') as f:
                data = f.read()
        data = data.translate(str.maketrans(
            {"！": "！\n", "：": "：\n", "。": "。\n"}))
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        filename_prefix_list = [
            item for item in file_path.relative_to(source_folder_path).parts if item != source_folder_path.name
        ]
        file_name_prefix = "_".join(x for x in filename_prefix_list if x)
        cut_file_name = f"{file_name_prefix}_{file}" if file_name_prefix else file
        cut_file_path = target_folder_path / cut_file_name
        with open(cut_file_path, "w", encoding='utf-8') as f:
            f.write(data)

loader = DirectoryLoader(utils.Gen_Data.Target_Path, glob='**/*.txt')
docs = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=int(utils.Gen_Data.Size), chunk_overlap=int(utils.Gen_Data.Overlay), separator='\n')
doc_texts = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='')
embeddings.client = sentence_transformers.SentenceTransformer(
    utils.Gen_Data.Model_Path, device=utils.Gen_Data.Device)
texts = [d.page_content for d in doc_texts]
metadatas = [d.metadata for d in doc_texts]
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
print("处理完成")
try:
    vectorstore_old = FAISS.load_local(
        'vectorstore_path', embeddings=embeddings)
    print("合并至已有索引")
    vectorstore_old.merge_from(vectorstore)
    vectorstore_old.save_local('vectorstore_path')
except:
    print("新建索引")
    vectorstore.save_local('vectorstore_path')
print("保存完成")
