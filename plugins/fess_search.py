import re
from utils.base import BaseSearch
from utils import utils
import jieba

with open("plugins/stopword.txt", encoding="utf-8") as f:
    stopwords = f.read().split('\n')


class FessSearch(BaseSearch):
    def __init__(self):
        super().__init__()

    def replace_strong(self, s):
        s = re.sub(r'<strong>', "", s)
        s = re.sub(r'</strong>', "", s)
        return s

    def remove_stopwords(self, search_query):
        search_query_without_stopwords = []
        for i in search_query:
            try:
                stopwords.index(i)
            except:
                search_query_without_stopwords.append(i)
        return search_query_without_stopwords

    def find(self, search_query):
        try:
            search_query = jieba.cut(search_query)
            search_query = self.remove_stopwords(search_query)
            search_query = " ".join(search_query)
            utils.logger.info(f"关键词: {search_query}")
            fess_path = utils.Fess["path"]
            url = f"http://{fess_path}/json/?q={search_query}&num=10&sort=score.desc&lang=zh_CN"
            res = self.session.get(
                url, headers=self.headers, proxies=self.proxies)
            r = res.json()
            r = r["response"]['result']
            return [{'title': r[i]['title'], 'content': self.replace_strong(r[i]['content_description'])}
                    for i in range(min(int(utils.Fess["count"]), len(r)))]
        except Exception as e:
            utils.logger.error(f"fess读取失败:{e}")
            return []


fess_search = FessSearch()
