from flask import Blueprint, request, Response, abort
import torch
import json
import os
from utils import utils

chat_ops = Blueprint('chat_ops', __name__)


@chat_ops.route('/chat/completions', methods=['POST'])
def request_completions():
    """
    处理聊天请求并返回事件流。

    请求数据:
        一个 JSON 对象，包含聊天参数:
        - max_tokens: 一个整数，表示生成文本的最大长度。
        - top_p: 一个浮点数，表示采样概率。
        - temperature: 一个浮点数，表示生成文本的多样性。
        - mix: 一个布尔值，表示是否混合多个模型的输出。
        - message: 一个数组，表示聊天历史记录。每个元素都是一个对象，包含一个名为 "role" 的字段，表示发言者的角色。
        示例: {"max_tokens": 2048, "top_p": 0.2, "temperature": 0.8, "mix":true, "message": [{"role":"user"}]}

    返回值:
        一个事件流，包含聊天响应。每个事件都是一个 JSON 对象，包含以下字段：
        - response: 一个字符串，表示聊天模型的响应文本。
    """
    max_length, top_p, temperature, mix, messages, prompt = get_request_params()
    # 初始化聊天模型
    history_formatted = utils.Model.chat_init(messages)

    def event_stream():
        error = ''
        response_text = ''
        # 获取 用户信息
        #IP = request.environ.get(
        #    'HTTP_X_REAL_IP') or request.environ.get('REMOTE_ADDR')
        with utils.mutex:
            #yield json.dumps({"response": (str(len(prompt))+'字正在计算')})
            yield "data: %s\n\n" %json.dumps({"response": (str(len(prompt))+'字正在计算')})
            utils.logger.info(f"\033[1;32mMessage:\033[1;31m{prompt}\033[1;37m")
            try:
                for response_text in utils.Model.chat(prompt, history_formatted, max_length, top_p, temperature, mix=mix):
                    if (response_text):
                        # yield "data: %s\n\n" %response_text
                        yield "data: %s\n\n" %json.dumps({"response": response_text})
                        #yield json.dumps({"response": response_text})
                yield "data: %s\n\n" %"[DONE]"
                #yield json.dumps({"response": response_text})
            except Exception as e:
                error = str(e)
                utils.logger.error(f"错误{utils.Red}{error}{utils.White}")
                response_text = ''
            torch.cuda.empty_cache()
        if response_text == '':
            yield "data: %s\n\n" %json.dumps({"response": ("发生错误，正在重新加载模型"+error)})
            #yield json.dumps({"response": (f"发生错误，正在重新加载模型{error}")})
    response = Response(event_stream(), content_type="text/event-stream")
    response.headers['Connection'] = 'keep-alive'
    response.headers['Cache-Control'] = 'no-cache'
    return response


def get_request_params():
    data = request.json
    if not data:
        abort(400, 'Missing request data')
    max_length = data.get('max_tokens', 2048)
    top_p = data.get('top_p', 0.2)
    temperature = data.get('temperature', 0.8)
    mix = data.get('mix', True)
    messages = data.get('messages')
    prompt = messages[-1]['content']
    return max_length, top_p, temperature, mix, messages, prompt
