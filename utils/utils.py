class UniveralSearchClass:
    def __init__(self, Fess=None, Bing=None):
        self.Fess = Fess
        self.Bing = Bing


class GenDataClass:
    def __init__(self, LibraryPath='', Target_Path='', Size=0, Overlap=0, Count=0, Model_path='', Device=''):
        self.LibraryPath = LibraryPath
        self.Target_Path = Target_Path,
        self.Size = Size,
        self.Overlay = Overlap,
        self.Count = Count,
        self.Model_Path = Model_path,
        self.Device = Device


Red = "\033[1;32m"
Green = "\033[1;31m"
White = "\033[1;37m"
mutex = None
logger = None
LLM_Type = ''
Tokenizer = None
Model = None
Library = None
GLM = None
Llama = None
Weight = None
Lora = None
Embeddings = None
Vectorstore = None
UniveralSearch = UniveralSearchClass()
Gen_Data = GenDataClass()
SearchAgent = None
OpenAI = None
GraphDB = None
CmdbAgent = None


