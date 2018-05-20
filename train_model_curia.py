from database import table_docs, table_doc_contents
import helpers
from nlp import model_trainer
import os
from nlp.preprocessing_curia import CURIAContentProcessor

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:5]

content_generator = (table_doc_contents.get_doc_content(doc) for doc in docs)
sentences = CURIAContentProcessor(content_generator)

helpers.create_folder_if_not_exists('trained_models')
path = os.path.join('trained_models', helpers.setup_json['model_path'])
model_trainer.train_model(sentences, path)