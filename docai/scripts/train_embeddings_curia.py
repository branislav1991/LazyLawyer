import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.scripts.save_doc_embeddings import save_doc_embeddings_word2vec
from docai.scripts.save_doc_embeddings import save_doc_embeddings_lsi
import gensim
from itertools import chain
import os
import pickle

def train_fasttext_curia(min_count, epoch_num, embedding_dim, learning_rate):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['fasttext_path'])

    content_gen = chain.from_iterable(ContentGenerator(docs))
    contents = list(content_gen) # generate all contents at once

    print('Initializing and training model...')
    model = gensim.models.FastText(sentences=contents, iter=epoch_num, size=embedding_dim, window=3, sg=1, min_count=min_count, negative=5, workers=4, alpha=learning_rate)

    # save final version
    model.save(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings_word2vec('fasttext.pickle', model)

def train_word2vec_curia(min_count, epoch_num, embedding_dim, learning_rate):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])

    content_gen = chain.from_iterable(ContentGenerator(docs))
    contents = list(content_gen) # generate all contents at once

    print('Initializing and training model...')
    model = gensim.models.Word2Vec(sentences=contents, iter=epoch_num, size=embedding_dim, window=3, sg=1, min_count=min_count, negative=5, workers=4, alpha=learning_rate)

    # save final version
    model.wv.save_word2vec_format(model_path, binary=True)

    print('Saving document embeddings...')
    save_doc_embeddings_word2vec('word2vec.pickle', model)

def train_lsi_curia(embedding_dim):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['lsi_path'])

    content_gen = ContentGenerator(docs)
    contents = [list(chain.from_iterable(content)) for content in content_gen]

    print('Initializing and training model...')
    dictionary = gensim.corpora.Dictionary(contents)
    content_bow = [dictionary.doc2bow(content) for content in contents]
    tfidf = gensim.models.TfidfModel(content_bow)
    content_tfidf = tfidf[content_bow]

    model = gensim.models.LsiModel(content_tfidf, id2word=dictionary, num_topics=embedding_dim)

    # save final version
    dictionary.save(os.path.splitext(model_path)[0] + '_dict.bin')
    tfidf.save(os.path.splitext(model_path)[0] + '_tfidf.bin')
    model.save(model_path)

    print('Saving document embeddings...')
    save_doc_embeddings_lsi('lsi.pickle', model, dictionary, tfidf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train word2vec model on CURIA documents')
    parser.add_argument('--model', choices=['word2vec', 'fasttext', 'lsi'], default='word2vec', help='which model should be trained')
    parser.add_argument('--min_count', type=int, default=3, help='minimal word frequency')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate')

    args = parser.parse_args()
    if args.model == 'word2vec':
        train_word2vec_curia(args.min_count, args.num_epochs, args.embedding_dim, args.learning_rate)
    elif args.model == 'fasttext':
        train_fasttext_curia(args.min_count, args.num_epochs, args.embedding_dim, args.learning_rate)
    elif args.model == 'lsi':
        train_lsi_curia(args.embedding_dim)
    else:
        raise ValueError('Invalid model name')
