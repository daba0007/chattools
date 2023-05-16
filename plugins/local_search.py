
import numpy as np
import re
from utils.base import BaseSearch
from utils import utils


divider='\n'

class LocalSearch(BaseSearch):
    def get_doc_by_id(self, id):
        return utils.Vectorstore.docstore.search(utils.Vectorstore.index_to_docstore_id[id])

    def process_strings(self, A, C, B):
        # find the longest common suffix of A and prefix of B
        common = ""
        for i in range(1, min(len(A), len(B)) + 1):
            if A[-i:] == B[:i]:
                common = A[-i:]
        # if there is a common substring, replace one of them with C and concatenate
        if common:
            return A[:-len(common)] + C + B
        # otherwise, just return A + B
        else:
            return A + B

    def get_doc(self, id,score,step):
        doc = self.get_doc_by_id(id)
        final_content=doc.page_content
        if step > 0:
            for i in range(1, step+1):
                try:
                    doc_before=self.get_doc_by_id(id-i)
                    if doc_before.metadata['source']==doc.metadata['source']:
                        final_content=self.process_strings(doc_before.page_content,divider,final_content)
                except:
                    pass
                try:
                    doc_after=self.get_doc_by_id(id+i)
                    if doc_after.metadata['source']==doc.metadata['source']:
                        final_content=self.process_strings(final_content,divider,doc_after.page_content)
                except:
                    pass
        return {'title': doc.metadata['source'],'content':re.sub(r'\n+', "\n", final_content)}

    def find(self, s,step = 1):
        try:
            embedding = utils.Vectorstore.embedding_function(s)
            scores, indices = utils.Vectorstore.index.search(np.array([embedding], dtype=np.float32), int(utils.Gen_Data.Count))
            docs = []
            for j, i in enumerate(indices[0]):
                if i == -1:
                    continue
                docs.append(self.get_doc(i,scores[0][j],step))
            return docs
        except Exception as e:
            utils.logger.error(e)
            return []
        
    def find_with_str(self, s, embedding=None, scores=None, step=1):
        try:
            if not embedding:
                embedding = utils.Vectorstore.embedding_function(s)
            if not scores:
                scores, indices = utils.Vectorstore.index.search(np.array([embedding], dtype=np.float32), int(utils.Gen_Data.Count))

            docs = [self.get_doc(i, scores[0][j], step) for j, i in enumerate(indices[0]) if i != -1]
            contentlist = [i["content"] for i in docs]
            return ' '.join(contentlist)
        except Exception as e:
            utils.logger.error(e)
            return ""




local_search = LocalSearch()