from plugins.search import find
from utils import utils
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate


class Glm6BChatBot(LLM):

    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    history_len: int = 10
    model: object = None
    tokenizer: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        device, _ = utils.GLM["strategy"].split()
        import torch
        if device == 'cuda' and torch.cuda.is_available():
            with torch.cuda.device(f"{device}:0"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history+[[None, response]]
        return response

    def chat_init(self, history):
        history_formatted = []
        current_chat = []
        for chat in history:
            if chat['role'] == "user":
                current_chat.append(chat['content'])
            elif chat['role'] in ("AI", "assistant"):
                current_chat.append(chat['content'])
                history_formatted.append(tuple(current_chat))
                current_chat = []
        return history_formatted

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            utils.GLM["path"], local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            utils.GLM["path"], local_files_only=True, trust_remote_code=True)
        if not (utils.Lora == '' or utils.Lora == None):
            print('Lora模型地址', utils.Lora)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                utils.Lora, local_files_only=True, trust_remote_code=True)

        device, precision = utils.GLM["strategy"].split()
        self.handle_device(precision, device)
        self.handle_precision(precision, device)
        self.model = self.model.eval()

    def handle_device(self, precision, device):
        if device == 'cpu':
            pass
        elif device == 'cuda':
            import torch
            if not (precision.startswith('fp16i') and torch.cuda.get_device_properties(0).total_memory < 1.4e+10):
                self.model = self.model.cuda()
        else:
            print('Error: 不受支持的设备')
            exit()

    def handle_precision(self, precision, device):
        if precision == 'fp16':
            self.model = self.model.half()
        elif precision == 'fp32':
            self.model = self.model.float()
        else:
            print('Error: 不受支持的精度')
            exit()
        if device == 'cuda':
            self.model = self.model.cuda()

    def chat(self, prompt, history_formatted=history, max_length=max_token, top_p=top_p, temperature=temperature, library="mix", step=1):
        search_results = find(prompt, library, step=step)
        print( f"搜索结果{search_results}")
        question = prompt
        prompt = ' '.join([result['content']
                          for result in search_results])
        if library == 'local':
            message_prompt=PromptTemplate(
                template="你现在只能准确的回答出信息,知道就说信息,不知道就说不知道\n已知信息:{information} 问题:{question}.",
                input_variables=["information", "question"],
            )
            prompt = message_prompt.format(information=prompt, question=question)
        else:
            message_prompt=PromptTemplate(
                template="你现在只能准确的回答出信息,限制在 100 字以内回答\n已知信息:{information} 问题:{question}.",
                input_variables=["information", "question"],
            )
            prompt = message_prompt.format(information=prompt, question=question)
        for response, _ in self.model.stream_chat(self.tokenizer, prompt, history_formatted,
                                                  max_length=max_length, top_p=top_p, temperature=temperature):
            yield response

    def chat_chain(self, prompt, history_formatted=history, max_length=max_token, top_p=top_p, temperature=temperature):
        for response, _ in self.model.stream_chat(self.tokenizer, prompt, history_formatted,
                                                  max_length=max_length, top_p=top_p, temperature=temperature):
            yield response


model = Glm6BChatBot()
