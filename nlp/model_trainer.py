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

class ContentProcessor():
    """This class processes content one-by-one by applying
    various preprocessing steps such as tokenization etc.
    It requires a content generator as input.
    """
    def __init__(self, content_generator):
        self.content_generator = content_generator
        self.pos_stoplist = set('PUNCT SPACE SYM NUM'.split())
        self.stoplist = set('for a of the and to in'.split())

    def __next__(self):
        content = next(self.content_generator)

        print('Preprocessing document...')

        # normalize and tokenize content
        content = normalize_text(content)
        tokens = nlp(content)

        # remove punctuations and symbols
        tokens = [x.text for x in tokens if x.pos_ not in self.pos_stoplist]

        # remove uninformational words
        tokens = [x for x in tokens if x not in self.stoplist]

        return tokens

    def __iter__(self):
        return self

def train_model(content_generator, save_path):
    """Trains a Word2Vec model using the contents supplied by
    content_generator. content_generator has to be an infinite
    generator. num_contents specifies the number of contents to
    determine the size of each epoch. The model gets saved in
    save_path after training.
    """
    tokens = ContentProcessor(content_generator)
    print('Training Word2Vec model...')
    model = models.Word2Vec(tokens, size=200, negative=10, window=5, min_count=5, workers=4, iter=20)
    model.save(save_path)