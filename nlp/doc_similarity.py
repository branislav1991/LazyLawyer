from collections import defaultdict
from gensim import corpora, models
import numpy as np
from nlp.preprocessing import normalize_text
import os
import regex
import spacy
from spacy.tokenizer import Tokenizer

model_path = os.path.join('nlp', 'word2vec_model.pickle')
model = models.Word2Vec.load(model_path)
nlp = spacy.load('en_core_web_sm')

def most_similar(content, topn):
    return model.most_similar(positive=[content], topn=topn)

def similarity(content1, content2):
    content1 = normalize_text(content1)
    content2 = normalize_text(content2)

    tokens1 = nlp(content1)
    tokens2 = nlp(content2)

    # remove punctuations and symbols
    pos_stoplist = set('PUNCT SPACE SYM'.split())
    tokens1 = [x for x in tokens1 if x.pos_ not in pos_stoplist]
    tokens2 = [x for x in tokens2 if x.pos_ not in pos_stoplist]

    # convert tokens to words
    tokens1 = [x.text for x in tokens1]
    tokens2 = [x.text for x in tokens2]

    # remove uninformational words
    stoplist = set('for a of the and to in'.split())
    tokens1 = [x for x in tokens1 if x not in stoplist]
    tokens2 = [x for x in tokens2 if x not in stoplist]

    doc1avg = np.mean(np.array([model[x] for x in tokens1 if x in model.wv.vocab]), axis=0)
    doc2avg = np.mean(np.array([model[x] for x in tokens2 if x in model.wv.vocab]), axis=0)

    doc1avg = doc1avg / np.linalg.norm(doc1avg)
    doc2avg = doc2avg / np.linalg.norm(doc2avg)

    similarity = np.dot(doc1avg, doc2avg)
    return similarity