from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
import os

save_path = os.path.join('trained_models', helpers.setup_json['model_path'])
model = Word2Vec(save_path)
model.load()

sim1 = model.word_similarity('judgment', 'court')
sim2 = model.word_similarity('judgment', 'company')
print('Similarity 1: {0}, Similarity 2: {1}'.format(sim1, sim2))
