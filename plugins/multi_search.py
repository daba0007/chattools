from bing_search import BingSearch
from fess_search import FessSearch

class MultiSearch:
    def __init__(self, weights):
        self.weights = weights
        self.searches = {
            'bing': BingSearch(),
            'fess': FessSearch()
        }

    def find(self, keyword):
        results = []
        for search_type, weight in self.weights.items():
            search = self.searches[search_type]
            search_results = search.find(keyword)
            results.extend(search_results[:int(len(search_results) * weight)])
        return results