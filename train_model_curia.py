from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.preprocessing_curia import CURIAContentProcessor
import os

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:500]

content_generator = (table_doc_contents.get_doc_content(doc) for doc in docs)
sentences = CURIAContentProcessor(content_generator)

helpers.create_folder_if_not_exists('trained_models')
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

model = Word2Vec(save_path)
model.init_and_save_vocab(sentences)
model.train() # train and save embedding matrices