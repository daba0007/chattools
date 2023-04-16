from plugins.bing_search import BingSearch
from plugins.multi_search import MultiSearch
from utils import utils

def find(prompt, mix=False):
    if mix:
       return  MultiSearch(utils.Weight).find(prompt)
    return BingSearch().find(prompt)
 