import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai.nlp.helpers import get_embedding_doc
from docai import helpers
import gensim
import os
import pickle

def save_doc_embeddings(file_name, model):
    """Saves document embeddings in a file
    using the provided model.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('saved_embeddings')
    embs = []

    for doc, content in zip(docs, content_gen):
        emb = get_embedding_doc(content, model)
        embs.append({'doc_id': doc['id'], 'emb': emb})

    with open(os.path.join('saved_embeddings', file_name), 'wb') as f:
        pickle.dump(embs, f)

def save_word2vec_curia(model_path, num_words):
    # load pretrained model
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', model_path)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=num_words)

    print('Saving document embeddings...')
    save_doc_embeddings('word2vec.pickle', model)

def save_fasttext_curia(model_path):
    # load pretrained model
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', model_path)
    model = gensim.models.FastText.load(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings('fasttext.pickle', model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save document embeddings using a certain model')
    parser.add_argument('model', choices=['word2vec', 'fasttext'], help='model for doc embeddings')
    parser.add_argument('model_path', help='model path')
    parser.add_argument('--num_words', type=int, default=1000000, help='max vocabulary size for word2vec')

    args = parser.parse_args()

    if args.model == 'word2vec':
        save_word2vec_curia(args.model_path, args.num_words)
    elif args.model == 'fasttext':
        save_fasttext_curia(args.model_path)
