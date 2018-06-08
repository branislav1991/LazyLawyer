from docai.database import table_doc_contents
from docai.nlp.curia_preprocessor import preprocess
from docai.nlp import phrases

class ContentGenerator:
    """Yields a document content generator based on the list of 
    documents. Yields in one iteration one document consisting of
    sentences of words.
    """
    def __init__(self, docs):
        self.docs = docs
        self.doc_gen = None

    def __iter__(self):
        self.doc_gen = (doc for doc in self.docs)
        return self
    
    def __next__(self):
        doc = table_doc_contents.get_doc_content(next(self.doc_gen))
        doc = preprocess(doc)
        doc = phrases.build_phrases_regex(doc)
        return doc
