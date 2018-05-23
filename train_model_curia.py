from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.curia_tokenizer import tokenize
import os

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:10]

helpers.create_folder_if_not_exists('trained_models')
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

model = Word2Vec()

document_gen = (tokenize(table_doc_contents.get_doc_content(doc)) for doc in docs)
model.init_and_save_vocab(document_gen, save_path)

document_gen = (tokenize(table_doc_contents.get_doc_content(doc)) for doc in docs)
model.init_and_save_idf(document_gen, save_path)

document_gen = (tokenize(table_doc_contents.get_doc_content(doc)) for doc in docs)
model.train(document_gen, save_path)