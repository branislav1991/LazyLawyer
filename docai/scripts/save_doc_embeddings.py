import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai.models.word2vec import Word2Vec
from docai.models.fasttext import FastText
from docai.models.load_word2vec_binary import load_word2vec_binary
from docai.nlp.vocabulary import Vocabulary, FastTextVocabulary
from docai import helpers
import os
import pickle

def save_doc_embeddings(file_name, vocabulary, model, strategy='average'):
    """Saves document embeddings in a file
    using the provided model and vocabulary. This function requires that the model
    supports the get_embedding_doc interface.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('saved_embeddings')
    embs = []

    for doc, content in zip(docs, content_gen):
        emb = model.get_embedding_doc(content, strategy=strategy)
        embs.append({'doc_id': doc['id'], 'emb': emb})

    with open(os.path.join('saved_embeddings', file_name), 'wb') as f:
        pickle.dump(embs, f)

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
    save_doc_embeddings('word2vec_pretrained.pickle', vocabulary, model)

def save_word2vec_curia():
    model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    print('Loading vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.load(vocab_path)

    print('Loading model...')
    model = Word2Vec(vocabulary)
    model.load(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings('word2vec.pickle', vocabulary, model)

def save_fasttext_curia():
    model_path = os.path.join('trained_models', helpers.setup_json['fasttext_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    print('Loading vocabulary...')
    vocabulary = FastTextVocabulary()
    vocabulary.load(vocab_path)

    print('Loading model...')
    model = FastText(vocabulary)
    model.load(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings('fasttext.pickle', vocabulary, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save document embeddings using a certain model')
    parser.add_argument('model', choices=['word2vec', 'word2vec_pretrained', 'fasttext'], default='word2vec', help='which model should be used')
    parser.add_argument('--num_words', type=int, default=200000, help='size of the vocabulary for pretrained embeddings')

    args = parser.parse_args()

    if args.model == 'word2vec_pretrained':
        save_word2vec_pretrained_curia(args.num_words)
    elif args.model == 'word2vec':
        save_word2vec_curia()
    elif args.model == 'fasttext':
        save_fasttext_curia()
