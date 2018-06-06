from docai.database import table_docs, table_doc_contents
from docai import helpers
from itertools import chain
from docai.nlp.curia_preprocessor import preprocess
from docai.nlp.word2vec_model import Word2Vec
from docai.nlp.vocabulary import Vocabulary
from docai.nlp import phrases
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
        self.doc_gen = (doc for doc in self.docs)
        return self
    
    def __next__(self):
        doc = table_doc_contents.get_doc_content(next(self.doc_gen))
        doc = preprocess(doc)
        return doc

def main():
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_name('Judgment')
    docs = docs[:100]
    document_gen = DocGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

    print('Initializing phrases...')
    doc_phrases = [phrases.build_phrases_regex(doc) for doc in document_gen]

    print('Initializing vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.initialize_and_save_vocab(doc_phrases, save_path)
    print('Initializing idf weights...')
    vocabulary.initialize_and_save_idf(doc_phrases, save_path)

    print('Initializing model...')
    model = Word2Vec(vocabulary)
    print('Starting training...')
    model.train(doc_phrases, save_path)

if __name__ == '__main__':
    main()
