from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.curia_preprocessor import preprocess
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

rules = [r"articl \d+\w*",
         r"paragraph \d+\w*",
         r"law no",
         r"law no \d+\w*",
         r"direct \d+\w*",
         r"^((31(?!\ (feb(ruary)?|apr(il)?|june?|(sep(?=\b|t)t?|nov)(emb)?)))|((30|29)(?!\ feb(ruary)?))|(29(?=\ feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (jan(uary)?|feb(ruary)?|ma(r(ch)?|y)|apr(il)?|ju((li?)|(ne?))|aug(ust)?|oct(ob)?|(sep(?=\b|t)t?|nov|dec)(emb)?)$",
         r"^((31(?!\ (feb(ruary)?|apr(il)?|june?|(sep(?=\b|t)t?|nov)(emb)?)))|((30|29)(?!\ feb(ruary)?))|(29(?=\ feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (jan(uary)?|feb(ruary)?|ma(r(ch)?|y)|apr(il)?|ju((li?)|(ne?))|aug(ust)?|oct(ob)?|(sep(?=\b|t)t?|nov|dec)(emb)?)\ ((1[6-9]|[2-9]\d)\d{2})$"]

docs = table_docs.get_docs_with_name('Judgment')

doc1 = preprocess(table_doc_contents.get_doc_content(docs[9000]))
doc1 = build_phrases_regex(doc1, rules=rules)

doc2 = preprocess(table_doc_contents.get_doc_content(docs[100]))
doc2 = build_phrases_regex(doc2, rules=rules)

sim = model.doc_similarity(doc1, doc2, strategy='tf-idf')

print('Similarity: {0}'.format(sim))
