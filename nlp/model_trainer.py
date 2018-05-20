import helpers
from nlp.word2vec_model import Word2Vec
import numpy as np
from tqdm import tqdm

def train_model(sentences, save_path):
    """Trains a document similarity model using the tokens and sentences
    provided by tokens. The model is saved to save_path after
    training.
    """
    print('Training Word2Vec model...')
    model = Word2Vec(sentences, save_path)
    model.save_vocab()
    model.train() # train and save embedding matrices