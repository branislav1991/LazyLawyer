from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.curia_tokenizer import tokenize
from nlp.vocabulary import Vocabulary
from nlp.phrases import build_phrases_regex
import os

save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

print('Loading vocabulary...')
vocabulary = Vocabulary()
vocabulary.load(save_path)

print('Loading model...')
model = Word2Vec(vocabulary)
model.load(save_path)

rules = [r"article \d+\w*"]

docs = table_docs.get_docs_with_name('Judgment')
doc1 = tokenize(table_doc_contents.get_doc_content(docs[5000]))
doc1 = build_phrases_regex([doc1], rules=rules)[0]
doc2 = tokenize(table_doc_contents.get_doc_content(docs[1]))
doc2 = build_phrases_regex([doc2], rules=rules)[0]

sim = model.doc_similarity(doc1, doc2, strategy='tf-idf')

print('Similarity: {0}'.format(sim))
