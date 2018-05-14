from collections import defaultdict
import gc
from gensim import corpora, models
import numpy as np
from nlp.preprocessing import normalize_text
import regex
import spacy
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def train_model(contents, save_path):
    print('Normalizing text...')
    contents = [normalize_text(content) for content in contents]
    gc.collect()

    print('Running tokenizer...')
    tokens = [nlp(content) for content in tqdm(contents)]
    gc.collect()

    # remove punctuations and symbols
    print('Removing punctuations and symbols...')
    pos_stoplist = set('PUNCT SPACE SYM NUM'.split())
    tokens = [[x for x in content if x.pos_ not in pos_stoplist] for content in tokens]
    gc.collect()

    # convert tokens to words
    print('Converting tokens to words...')
    tokens = [[x.text for x in content] for content in tokens]
    gc.collect()

    # remove uninformational words
    print('Removing most frequent words...')
    stoplist = set('for a of the and to in'.split())
    tokens = [[x for x in content if x not in stoplist] for content in tokens]
    gc.collect()

    print('Training Word2Vec model...')
    model = models.Word2Vec(tokens, size=200, negative=10, window=5, min_count=5, workers=4, iter=2)
    model.save(save_path)