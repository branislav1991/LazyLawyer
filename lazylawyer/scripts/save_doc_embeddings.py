import argparse
from lazylawyer.database import table_docs
from lazylawyer.content_generator import ContentGenerator
from lazylawyer.nlp.helpers import get_embedding_doc_word2vec
from lazylawyer.nlp.helpers import get_embedding_doc_lsi
from lazylawyer import helpers
import gensim
from itertools import chain
import os
import pickle

def save_doc_embeddings_word2vec(file_name, model):
    """Saves document embeddings in a file
    using the provided word2vec or fasttext model.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('saved_embeddings')
    embs = []

    for doc, content in zip(docs, content_gen):
        emb = get_embedding_doc_word2vec(content, model, stopword_removal=True)
        embs.append({'doc_id': doc['id'], 'emb': emb})

    with open(os.path.join('saved_embeddings', file_name), 'wb') as f:
        pickle.dump(embs, f)

def save_doc_embeddings_doc2vec(file_name, model):
    """Saves document embeddings in a file
    using the provided doc2vec model.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)
    contents = [list(chain.from_iterable(content)) for content in content_gen]

    helpers.create_folder_if_not_exists('saved_embeddings')
    embs = []

    for doc, content in zip(docs, contents):
        emb = model.infer_vector(content)
        embs.append({'doc_id': doc['id'], 'emb': emb})

    with open(os.path.join('saved_embeddings', file_name), 'wb') as f:
        pickle.dump(embs, f)

def save_doc_embeddings_lsi(file_name, model, dictionary, tfidf):
    """Saves document embeddings in a file
    using the provided lsi model.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)
    contents = [list(chain.from_iterable(content)) for content in content_gen]

    helpers.create_folder_if_not_exists('saved_embeddings')
    embs = []

    for doc, content in zip(docs, contents):
        emb = get_embedding_doc_lsi(content, model, dictionary, tfidf)
        embs.append({'doc_id': doc['id'], 'emb': emb})

    with open(os.path.join('saved_embeddings', file_name), 'wb') as f:
        pickle.dump(embs, f)

def save_word2vec_curia(model_path, num_words):
    # load pretrained model
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', model_path)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=num_words)

    print('Saving document embeddings...')
    save_doc_embeddings_word2vec('word2vec.pickle', model)

def save_fasttext_curia(model_path, pretrained):
    # load pretrained model
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', model_path)

    if pretrained == False:
        model = gensim.models.FastText.load(model_path)
    else:
        model = gensim.models.FastText.load_fasttext_format(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings_word2vec('fasttext.pickle', model)

def save_doc2vec_curia(model_path):
    # load pretrained model
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', model_path)
    model = gensim.models.Doc2Vec.load(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings_doc2vec('doc2vec.pickle', model)

def save_lsi_curia(model_path):
    # load pretrained model
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', model_path)
    model = gensim.models.LsiModel.load(model_path)
    dictionary = gensim.corpora.Dictionary.load(os.path.splitext(model_path)[0] + '_dict.bin')
    tfidf = gensim.models.TfidfModel.load(os.path.splitext(model_path)[0] + '_tfidf.bin')

    print('Saving document embeddings...')
    save_doc_embeddings_lsi('lsi.pickle', model, dictionary, tfidf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save document embeddings using a certain model')
    parser.add_argument('model', choices=['word2vec', 'fasttext', 'fasttext_pretrained', 'lsi'], help='model for doc embeddings')
    parser.add_argument('model_path', help='model path')
    parser.add_argument('--num_words', type=int, default=1000000, help='max vocabulary size for word2vec')

    args = parser.parse_args()

    if args.model == 'word2vec':
        save_word2vec_curia(args.model_path, args.num_words)
    elif args.model == 'fasttext':
        save_fasttext_curia(args.model_path, pretrained=False)
    elif args.model == 'fasttext_pretrained':
        save_fasttext_curia(args.model_path, pretrained=True)
    elif args.model == 'doc2vec':
        save_doc2vec_curia(args.model_path)
    elif args.model == 'lsi':
        save_lsi_curia(args.model_path)
