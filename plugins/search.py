from plugins.bing_search import bing_search
from plugins.multi_search import MultiSearch
from plugins.fess_search import fess_search
from plugins.local_search import local_search
from utils import utils

search_functions = {
    'mix': MultiSearch(utils.Weight).find,
    'bing': bing_search.find,
    'fess': fess_search.find,
    'local': local_search.find
}

def find(prompt, library='mix', step=1):
    return search_functions[library](prompt, step)
