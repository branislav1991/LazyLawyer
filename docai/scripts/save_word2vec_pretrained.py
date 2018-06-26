import argparse
from docai import helpers
from docai.models.word2vec import Word2Vec
from docai.models.load_word2vec_binary import load_word2vec_binary
from docai.nlp.vocabulary import Vocabulary
from docai.save_doc_embeddings import save_doc_embeddings
import os

def save_word2vec_pretrained_curia(num_words):
    # load pretrained model from Google word2vec embeddings
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', helpers.setup_json['googlenews_word2vec_path'])
    word_dict = load_word2vec_binary(model_path, max_vectors=num_words)

    print('Creating vocabulary...')
    vocabulary = Vocabulary()
    idx_dict = vocabulary.load_from_dict(word_dict)

    print('Loading model...')
    model = Word2Vec(vocabulary, embedding_dim=300)
    model.load_from_dict(idx_dict)

    print('Saving document embeddings...')
    save_doc_embeddings(vocabulary, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run whole crawling pipeline up to document content saving')
    parser.add_argument('--num_words', type=int, default=200000, help='size of the vocabulary')

    args = parser.parse_args()
    save_word2vec_pretrained_curia(args.num_words)
