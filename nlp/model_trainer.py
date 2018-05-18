from gensim import corpora, models
import helpers
import numpy as np
from tqdm import tqdm

def train_model(tokens, save_path):
    """Trains a document similarity model using the tokens and sentences
    provided by tokens. The model is saved to save_path after
    training.
    """
    print('Training Word2Vec model...')
    model = models.Word2Vec(tokens, size=100, window=4, min_count=3, workers=4, iter=100)
    model.save(save_path)