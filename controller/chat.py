from flask import Blueprint, request, Response, abort, stream_with_context
import torch
import json
import os
from utils import utils

chat_ops = Blueprint('chat_ops', __name__)

@chat_ops.route('/chat/completions', methods=['POST'])
def request_completions():
    max_length, top_p, temperature, library, messages, prompt = get_request_params()
    history_formatted = utils.Model.chat_init(messages)
    def event_stream():
        error = ''
        response_text = ''
        with utils.mutex:
            #yield "%s\n\n" % json.dumps({"response": (str(len(prompt))+'字正在计算')})
            utils.logger.info(f"\033[1;32mMessage:\033[1;31m{prompt}\033[1;37m")
            try:
                for response_text, history in utils.Model.chat(prompt, history_formatted, max_length, top_p, temperature, library):
                    if (response_text):
                        # yield "data: %s\n\n" %response_text
                        yield "%s\n\n" % json.dumps({"response": response_text, "history": history})
                yield "%s\n\n" % json.dumps({"response": "[DONE]"})
            except Exception as e:
                error = str(e)
                utils.logger.error(f"错误{utils.Red}{error}{utils.White}")
                response_text = ''
            torch.cuda.empty_cache()
        if response_text == '':
            yield "%s\n\n" % json.dumps({"response": f"发生错误，正在重新加载模型{error}"})
    response = Response(stream_with_context(event_stream()), content_type="text/event-stream")
    response.headers['Connection'] = 'keep-alive'
    response.headers['Cache-Control'] = 'no-cache'
    return response


def get_request_params():
    data = request.json
    if not data:
        abort(400, 'Missing request data')
    max_length = data.get('max_tokens', 10000)
    top_p = data.get('top_p', 0.2)
    temperature = data.get('temperature', 0.8)
    library = data.get('library', 'mix')
    messages = data.get('messages')
    prompt = messages[-1]['content']
    return max_length, top_p, temperature, library, messages, prompt
