import requests

class BaseSearch:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
        self.proxies = {"http": None,"https": None,}

    def find(self, search_query=""):
        pass

class ChatBot:
    def __init__(self, model):
        self.model = model
        
    def chat_init(self, history):
        pass
    
    def chat(self, prompt, history_formatted, max_length, top_p, temperature, mix=False):
        pass
    
    def load_model(self):
        pass

	