from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.curia_sentence_processor import CURIASentenceProcessor
import os

save_path = os.path.join('trained_models', helpers.setup_json['model_path'])
model = Word2Vec()
model.load(save_path)


docs = table_docs.get_docs_with_name('Judgment')
doc1 = table_doc_contents.get_doc_content(docs[10])
doc2 = table_doc_contents.get_doc_content(docs[1000])

sentences1 = CURIAContentProcessor([doc1])
sentences2 = CURIAContentProcessor([doc2])

sim = model.doc_similarity(sentences1, sentences2)

print('Similarity: {0}'.format(sim))
