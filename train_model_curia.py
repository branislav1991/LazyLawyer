from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.curia_sentence_processor import CURIASentenceProcessor
from nlp.curia_document_processor import CURIADocumentProcessor
import os

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')

content_generator_sent = (table_doc_contents.get_doc_content(doc) for doc in docs)
sentence_gen = CURIASentenceProcessor(content_generator_sent)

content_generator_doc = (table_doc_contents.get_doc_content(doc) for doc in docs)
document_gen = CURIADocumentProcessor(content_generator_doc)

helpers.create_folder_if_not_exists('trained_models')
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

model = Word2Vec()
model.init_and_save_vocab(sentence_gen, save_path)
model.init_and_save_idf(document_gen, save_path)
model.train(save_path) # train and save embedding matrices