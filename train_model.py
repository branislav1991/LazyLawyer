from database import table_docs, table_doc_contents
from nlp import model_trainer

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')
contents = [table_doc_contents.get_doc_content(doc) for doc in docs]

print('Beginning training...')
model_trainer.train_model(contents, 'word2vec_model.pickle')
print('Finished!')