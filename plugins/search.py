from plugins.bing_search import BingSearch
from plugins.fess_search import FessSearch
from plugins.multi_search import MultiSearch
from utils import utils


if utils.Library_type == 'mix':
    library = MultiSearch(utils.Weight)
elif utils.Library_type == 'fess':
    library = FessSearch()
elif utils.Library_type == 'bing':
    library = BingSearch()
else:
    raise ValueError(f"Invalid search type: {utils.Library_type}")

def find(prompt):
	return library.find(prompt)
 