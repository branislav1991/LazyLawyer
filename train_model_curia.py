from database import table_docs, table_doc_contents
import helpers
from nlp.curia_tokenizer import tokenize
from nlp.word2vec_model import Word2Vec
import os

class DocGenerator:
    """Yields a document content generator based on the list of 
    documents.
    """
    def __init__(self, docs):
        self.docs = docs

    def __iter__(self):
        doc_gen = (tokenize(table_doc_contents.get_doc_content(doc)) for doc in self.docs)
        return doc_gen

print("Initializing database and loading documents...")
docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:10]
document_gen = DocGenerator(docs)

helpers.create_folder_if_not_exists('trained_models')
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

print('Initializing model...')
model = Word2Vec()
print('Initializing vocabulary...')
model.init_and_save_vocab(document_gen, save_path)
print('Initializing idf weights...')
model.init_and_save_idf(document_gen, save_path)
print('Starting training...')
model.train(document_gen, save_path)