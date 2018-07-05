import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.models.word2vec import Word2Vec
from docai.models.fasttext import FastText
from docai.models.elmo import ELMo
from docai.models.vocabulary import Vocabulary, FastTextVocabulary
from docai.scripts.save_doc_embeddings import save_doc_embeddings
import os
import pickle

def train_elmo_curia():
    raise NotImplementedError()
    # print("Initializing database and loading documents...")
    # docs = table_docs.get_docs_with_names(['Judgment'])
    # content_gen = ContentGenerator(docs)

    # helpers.create_folder_if_not_exists('trained_models')
    # model_path = os.path.join('trained_models', helpers.setup_json['elmo_path'])
    # vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    # contents = list(content_gen) # generate all contents at once

    # print('Initializing vocabulary...')
    # vocabulary = Vocabulary()
    # vocabulary.initialize_and_save_vocab(contents, vocab_path)
    # print('Initializing idf weights...')
    # vocabulary.initialize_and_save_idf(contents, vocab_path)

    # print('Initializing model...')
    # model = ELMo(vocabulary)
    # print('Starting training...')
    # model.train(contents, model_path)

    # print('Saving document embeddings...')
    # save_doc_embeddings('elmo.pickle', vocabulary, model)

def train_fasttext_curia(num_words, num_ngrams, max_ngram, epoch_num, embedding_dim, learning_rate):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['fasttext_path'])
    meta_path = model_path + '_meta.pickle'

    contents = list(content_gen) # generate all contents at once

    print('Initializing metainfo...')
    vocabulary = FastTextVocabulary(documents=contents, max_vocab_size=num_words, max_ngram_size=num_ngrams, max_ngram=max_ngram)

    meta_info = dict()
    meta_info['embedding_dim'] = embedding_dim
    meta_info['max_ngram'] = max_ngram
    meta_info['word_count'] = vocabulary.get_count() # the word count contains all information

    with open(meta_path, 'wb') as f:
        pickle.dump(meta_info, f)

    print('Initializing model...')
    model = FastText(vocabulary, embedding_dim=embedding_dim)
    print('Starting training...')
    model.train(contents, model_path, epoch_num=epoch_num, batch_size=16, window_size=3, neg_sample_num=5, learning_rate=learning_rate)

    # save final version
    model.save(model_path + '_final.pickle')

    print('Saving document embeddings...')
    save_doc_embeddings('fasttext.pickle', vocabulary, model)

def train_word2vec_curia(num_words, epoch_num, embedding_dim, learning_rate):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
    meta_path = model_path + '_meta.pickle'

    contents = list(content_gen) # generate all contents at once

    print('Initializing metainfo...')
    vocabulary = Vocabulary(documents=contents, max_vocab_size=num_words)

    meta_info = dict()
    meta_info['embedding_dim'] = embedding_dim
    meta_info['word_count'] = vocabulary.get_count() # the word count contains all information

    vocabulary.initialize_idf(contents)
    meta_info['idf'] = vocabulary.get_idf()

    with open(meta_path, 'wb') as f:
        pickle.dump(meta_info, f)

    print('Initializing model...')
    model = Word2Vec(vocabulary, embedding_dim=embedding_dim)
    print('Starting training...')
    model.train(contents, model_path, epoch_num=epoch_num, batch_size=16, window_size=3, neg_sample_num=5, learning_rate=learning_rate)

    # save final version
    model.save(model_path + '_final.pickle')

    print('Saving document embeddings...')
    save_doc_embeddings('word2vec.pickle', vocabulary, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train word2vec model on CURIA documents')
    parser.add_argument('--model', choices=['word2vec', 'fasttext', 'elmo'], default='word2vec', help='which model should be trained')
    parser.add_argument('--num_words', type=int, default=100000, help='size of the vocabulary')
    parser.add_argument('--num_ngrams', type=int, default=200000, help='size of the ngrams for fasttext')
    parser.add_argument('--max_ngram', type=int, default=10, help='maximal number of ngrams for each word')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate')

    args = parser.parse_args()
    if args.model == 'word2vec':
        train_word2vec_curia(args.num_words, args.num_epochs, args.embedding_dim, args.learning_rate)
    elif args.model == 'fasttext':
        train_fasttext_curia(args.num_words, args.num_ngrams, args.max_ngram, args.num_epochs, args.embedding_dim, args.learning_rate)
    elif args.model == 'elmo':
        train_elmo_curia()
