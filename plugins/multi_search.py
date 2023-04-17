from plugins.bing_search import bing_search
from plugins.fess_search import fess_search

class MultiSearch:
    def __init__(self, weights):
        self.weights = weights
        self.searches = {
            'bing': bing_search,
            'fess': fess_search
        }

    def find(self, keyword):
        results = []
        for search_type, weight in self.weights.items():
            search = self.searches[search_type]
            search_results = search.find(keyword)
            results.extend(search_results[:int(len(search_results) * weight)])
        return results