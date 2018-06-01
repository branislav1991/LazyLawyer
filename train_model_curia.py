from database import table_docs, table_doc_contents
import helpers
from itertools import chain
from nlp.curia_preprocessor import preprocess
from nlp.word2vec_model import Word2Vec
from nlp.vocabulary import Vocabulary
from nlp.phrases import build_phrases_regex
import os
import Stemmer

stemmer = Stemmer.Stemmer('english')

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
docs = docs[:5]
document_gen = DocGenerator(docs)

helpers.create_folder_if_not_exists('trained_models')
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

print('Initializing phrases...')
rules = [r"articl \d+\w*"]
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