from database import table_docs, table_doc_contents
from nlp import model_trainer
import os

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')

content_generator = (table_doc_contents.get_doc_content(doc) for doc in docs)

print('Beginning training...')
path = os.path.join('nlp', 'word2vec_model.pickle')
model_trainer.train_model(content_generator, path)
print('Finished!')