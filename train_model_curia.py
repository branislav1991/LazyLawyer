from database import table_docs, table_doc_contents
import helpers
from itertools import chain
from nlp.curia_preprocessor import preprocess
from nlp.word2vec_model import Word2Vec
from nlp.vocabulary import Vocabulary
from nlp.phrases import build_phrases_regex
import os

class DocGenerator:
    """Yields a document content generator based on the list of 
    documents. Yields in one iteration one document consisting of
    sentences of words.
    """
    def __init__(self, docs):
        self.docs = docs
        self.doc_gen = None

    def __iter__(self):
        self.doc_gen = (doc for doc in docs)
        return self
    
    def __next__(self):
        doc = table_doc_contents.get_doc_content(next(self.doc_gen))
        doc = preprocess(doc)
        return doc

print("Initializing database and loading documents...")
docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:100]
document_gen = DocGenerator(docs)

helpers.create_folder_if_not_exists('trained_models')
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

print('Initializing phrases...')
rules = [r"articl \d+\w*",
         r"paragraph \d+\w*",
         r"law no",
         r"law no \d+\w*",
         r"direct \d+\w*",
         r"^((31(?!\ (feb(ruary)?|apr(il)?|june?|(sep(?=\b|t)t?|nov)(emb)?)))|((30|29)(?!\ feb(ruary)?))|(29(?=\ feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (jan(uary)?|feb(ruary)?|ma(r(ch)?|y)|apr(il)?|ju((li?)|(ne?))|aug(ust)?|oct(ob)?|(sep(?=\b|t)t?|nov|dec)(emb)?)$",
         r"^((31(?!\ (feb(ruary)?|apr(il)?|june?|(sep(?=\b|t)t?|nov)(emb)?)))|((30|29)(?!\ feb(ruary)?))|(29(?=\ feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (jan(uary)?|feb(ruary)?|ma(r(ch)?|y)|apr(il)?|ju((li?)|(ne?))|aug(ust)?|oct(ob)?|(sep(?=\b|t)t?|nov|dec)(emb)?)\ ((1[6-9]|[2-9]\d)\d{2})$"]
phrases = [build_phrases_regex(doc, rules=rules) for doc in document_gen]

print('Initializing vocabulary...')
vocabulary = Vocabulary()
vocabulary.initialize_and_save_vocab(phrases, save_path)
print('Initializing idf weights...')
vocabulary.initialize_and_save_idf(phrases, save_path)

print('Initializing model...')
model = Word2Vec(vocabulary)
print('Starting training...')
model.train(phrases, save_path)