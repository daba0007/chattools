from plugins.search import find
from utils.base import ChatBot
from utils import utils

class Glm6BChatBot(ChatBot):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(model)
        self.tokenizer = tokenizer
        
    def chat_init(self, history):
        history_formatted = None
        if history is not None:
            history_formatted = []
            current_chat = []
            for _, old_chat in enumerate(history):
                if len(current_chat) == 0 and old_chat['role'] == "user":
                    current_chat.append(old_chat['content'])
                elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                    current_chat.append(old_chat['content'])
                    history_formatted.append(tuple(current_chat))
                    current_chat = []
                else:
                    continue
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
            self.model = PeftModel.from_pretrained(utils.Lora, local_files_only=True, trust_remote_code=True)

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
        elif precision.startswith('fp16i'):
            precision_bits = int(precision[5:])
            self.model = self.model.quantize(precision_bits)
            if device == 'cuda':
                self.model = self.model.cuda()
            self.model = self.model.half()
        elif precision.startswith('fp32i'):
            precision_bits = int(precision[5:])
            self.model = self.model.quantize(precision_bits)
            if device == 'cuda':
                self.model = self.model.cuda()
            self.model = self.model.float()
        else:
            print('Error: 不受支持的精度')
            exit()

    def chat(self, prompt, history_formatted, max_length, top_p, temperature, library="mix"):
        search_results = find(prompt, library)
        prompt = ' '.join([prompt] + [result['content'] for result in search_results])
        for response, _ in self.model.stream_chat(self.tokenizer, prompt, history_formatted,
                                                max_length=max_length, top_p=top_p, temperature=temperature):
            yield response
            
model = Glm6BChatBot()