import re
from plugins.textrank4zh import TextRank4Keyword
from utils.base import BaseSearch
from utils import utils

class FessSearch(BaseSearch):
    def __init__(self):
        super().__init__()
        self.tr4w = TextRank4Keyword()

    def replace_strong(self, s):
        s = re.sub(r'<strong>', "", s)
        s = re.sub(r'</strong>', "", s)
        return s

    def find(self, search_query):
        try:
            self.tr4w.analyze(text=search_query, lower=True, window=2) 
            search_query = ' '.join([i['word'] for i in self.tr4w.get_keywords(20, word_min_len=1)])
            print("关键词：", search_query)
            url = 'http://' + utils.Fess["path"] + '/json/?q={}&num=10&sort=score.desc&lang=zh_CN'.format(search_query)
            res = self.session.get(url, headers=self.headers, proxies=self.proxies)
            r = res.json()
            r = r["response"]['result']
            return [{'title': r[i]['title'], 'content': self.replace_strong(r[i]['content_description'])}
                    for i in range(min(int(utils.Fess["count"]), len(r)))]
        except Exception as e:
            print("fess读取失败", e)
            return []