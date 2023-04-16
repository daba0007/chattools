from bs4 import BeautifulSoup
from utils import utils
from utils.base import BaseSearch

class BingSearch(BaseSearch):
    def __init__(self):
        super().__init__()

    def find(self, search_query):
        url = 'https://cn.bing.com/search?q={}'.format(search_query)
        try:
            res = self.session.get(url, headers=self.headers ,proxies=self.proxies)
            # 解析响应并提取搜索结果
            soup = BeautifulSoup(res.text, 'html.parser')
            results = soup.find_all('li', class_='b_algo')
            search_results = []
            for result in results:
                title = result.find('h2').find('a').text
                link = result.find('h2').find('a')['href']
                content = result.find('div', class_='b_caption').find('p').text
                search_results.append({'title': "["+title+"]("+link+")", 'content': content})
            return search_results[:min(int(utils.Bing["count"]), len(search_results))]
        except Exception as e:
            print(f"Bing Search error occurred: {e}")
            return []