from flask import Flask
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import torch
from utils import utils
import yaml
from flask_cors import CORS

# SingleTon
app = Flask(__name__)

CORS(app, supports_credentials=True)


def setup_logger(app):
    # define log size

    log_size = 1024 * 1000 * 100 * 5
    if not os.path.exists('log'):
        os.mkdir('log')

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')

    info_file_handler = RotatingFileHandler(
        'log/info.log', maxBytes=log_size, backupCount=10, encoding='UTF-8')
    info_file_handler.setFormatter(formatter)
    info_file_handler.setLevel(logging.INFO)

    error_file_handler = RotatingFileHandler(
        'log/error.log', maxBytes=log_size, backupCount=10, encoding='UTF-8')
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)

    access_log = logging.getLogger('werkzeug')
    access_file_handler = RotatingFileHandler(
        'log/access.log', maxBytes=log_size, backupCount=10, encoding='UTF-8')
    access_file_handler.setFormatter(formatter)
    access_file_handler.setLevel(logging.INFO)
    access_log.addHandler(access_file_handler)
    access_log.propagate = False

    app.logger.addHandler(info_file_handler)
    app.logger.addHandler(error_file_handler)
    app.logger.setLevel(logging.INFO)


def load_LLM():
    try:
        from importlib import import_module
        utils.Model = import_module(f"plugins.llm_{utils.LLM_Type}").model
    except Exception as e:
        print(f"LLM模型加载失败:{e}")


def load_model():
    utils.mutex.acquire()
    utils.Model.load_model()
    utils.mutex.release()
    torch.cuda.empty_cache()
    print(utils.Green, "模型加载完成", utils.White)
    text = "你好,介绍下自己"
    utils.Model(text)


def setting(config):
    """
    更新 utils 模块中的变量。

    参数:
        config: 一个字典，包含配置信息。

    返回值:
        无。
    """
    try:
        utils.LLM_Type = config["llm_type"]
        utils.GLM = config["glm"]
        utils.Llama = config["llama"]
        utils.Weight = config["weight"]
        utils.Lora = config["lora"]
        utils.Fess = config["fess"]
        utils.Bing = config["bing"]
    except KeyError as e:
        raise ValueError(f'Missing key in config: {e}')

    if not isinstance(utils.LLM_Type, str):
        raise TypeError(
            f'LLM_Type must be a string, got {type(utils.LLM_Type)}')


# 读取环境变量
with open('config.yaml', 'r') as f:
    setting(yaml.safe_load(f))
# 设置日志
setup_logger(app)
utils.logger = app.logger


# Load Model
load_LLM()
utils.mutex = threading.Lock()

# Load Model
thread_load_model = threading.Thread(target=load_model)
thread_load_model.start()
