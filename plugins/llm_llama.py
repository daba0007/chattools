from search import find
from utils.base import ChatBot
from utils import utils

class LlamaChatBot(ChatBot):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        from llama_cpp import Llama
        self.model = Llama(model_path=utils.Llama["path"], use_mlock=True, n_ctx=4096)

    def chat_init(self, history):
        history_formatted = None
        if history is not None:
            history_formatted = ""
            for _, old_chat in enumerate(history):
                if old_chat['role'] == "user":
                    history_formatted += "Q: "+old_chat['content']+'\n'
                elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                    history_formatted += " A: "+old_chat['content']+'\n'
                else:
                    continue
        return history_formatted+" "

    def chat(self, prompt, history_formatted, max_length, top_p, temperature, mix=False):
        if mix:
            search_results = find(prompt)
            prompt = ' '.join([prompt] + [result['content'] for result in search_results])
            prompt = history_formatted+"%s\nAssistant: " % prompt
        else:
            prompt = history_formatted+"Human: %s\nAssistant: " % prompt
        stream = self.model(prompt,
                            stop=["Human:", "### Hum",], temperature=temperature,
                            max_tokens=max_length, top_p=top_p, stream=True)
        text = ""
        for output in stream:
            text += output["choices"][0]["text"]
            yield text