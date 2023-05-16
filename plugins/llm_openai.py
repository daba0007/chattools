from plugins.search import find
from utils import utils
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
import openai
from langchain.llms import OpenAI
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
openai.api_base = "https://api.chatanywhere.com.cn/v1"

class OpenAIChatBot(LLM):

    max_token: int = 1000
    temperature: float = 0
    top_p = 0.9
    history = []
    history_len: int = 3

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "OpenAI"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response = openai.Completion.create(
            engine=utils.OpenAI["engine"],
            prompt=prompt,
            max_tokens=self.max_token,
            n=1,
            stop=stop,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0
        )
        message = response.choices[0].text.strip()
        self.history.append([prompt, message])
        if len(self.history) > self.history_len:
            self.history.pop(0)
        return message

    def chat_init(self, history):
        history_formatted = []
        current_chat = []
        for chat in history:
            # 用户信息
            if chat['role'] == "user":
                current_chat.append(chat['content'])
            # 机器人信息
            elif chat['role'] in ("AI", "assistant"):
                current_chat.append(chat['content'])
                history_formatted.append(tuple(current_chat))
                current_chat = []
        return history_formatted

    def load_model(self):
        openai.api_key = utils.OpenAI["key"]
        import os
        os.environ["OPENAI_API_KEY"] = utils.OpenAI["key"]
            
    def chat(self, prompt, history_formatted=history, max_length=max_token, top_p=top_p, temperature=temperature, library="mix", step=1):
        search_results = find(prompt, library, step=step)
        print( f"搜索结果{search_results}")
        question = prompt
        prompt = ' '.join([result['content']
                          for result in search_results])
        if library == 'local':
            if history_formatted == []:
                message_prompt=PromptTemplate(
                    template="请从已知信息和消息历史中查找答案:\n已知信息:{information} \n问题:{question} \n你的回答是:",
                    input_variables=["information", "question"],
                )
                prompt = message_prompt.format(information=prompt, question=question)
            else:
                message_prompt=PromptTemplate(
                    template="已知信息:{information} 问题:{question}.你的回答:",
                    input_variables=["information", "question"],
                )
                prompt = message_prompt.format(information=prompt, question=question)
        else:
            if history_formatted == []:
                message_prompt=PromptTemplate(
                    template="请从已知信息和消息历史中查找答案:\n已知信息:{information} \n问题:{question} \n你的回答:",
                    input_variables=["information", "question"],
                )
                prompt = message_prompt.format(information=prompt, question=question)
            else:
                message_prompt=PromptTemplate(
                    template="已知信息:{information} 问题:{question}.你的回答:",
                    input_variables=["information", "question"],
                )
                prompt = message_prompt.format(information=prompt, question=question)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        yield response['choices'][0]['message']['content']


model = OpenAIChatBot()
