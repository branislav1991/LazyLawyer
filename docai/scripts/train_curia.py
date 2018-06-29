import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.models.word2vec import Word2Vec
from docai.models.fasttext import FastText
from docai.models.elmo import ELMo
from docai.nlp.vocabulary import Vocabulary, FastTextVocabulary
from docai.scripts.save_doc_embeddings import save_doc_embeddings
import os

def train_elmo_curia():
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['elmo_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    contents = list(content_gen) # generate all contents at once

    print('Initializing vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.initialize_and_save_vocab(contents, vocab_path)
    print('Initializing idf weights...')
    vocabulary.initialize_and_save_idf(contents, vocab_path)

    print('Initializing model...')
    model = ELMo(vocabulary)
    print('Starting training...')
    model.train(contents, model_path)

    print('Saving document embeddings...')
    save_doc_embeddings(vocabulary, model)

def train_fasttext_curia(num_ngrams, epoch_num):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['fasttext_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    contents = list(content_gen) # generate all contents at once

    print('Initializing vocabulary...')
    vocabulary = FastTextVocabulary(vocabulary_size=num_ngrams)
    vocabulary.initialize_and_save_vocab(contents, vocab_path)
    print('Initializing idf weights...')
    vocabulary.initialize_and_save_idf(contents, vocab_path)

    print('Initializing model...')
    model = FastText(vocabulary)
    print('Starting training...')
    model.train(contents, model_path, epoch_num=epoch_num)

    print('Saving document embeddings...')
    save_doc_embeddings(vocabulary, model)

def train_word2vec_curia(num_words, epoch_num):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    contents = list(content_gen) # generate all contents at once

    print('Initializing vocabulary...')
    vocabulary = Vocabulary(vocabulary_size=num_words)
    vocabulary.initialize_and_save_vocab(contents, vocab_path)
    print('Initializing idf weights...')
    vocabulary.initialize_and_save_idf(contents, vocab_path)

    print('Initializing model...')
    model = Word2Vec(vocabulary)
    print('Starting training...')
    model.train(contents, model_path, epoch_num=epoch_num)

    print('Saving document embeddings...')
    save_doc_embeddings(vocabulary, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train word2vec model on CURIA documents')
    parser.add_argument('--model', choices=['word2vec', 'fasttext', 'elmo'], default='word2vec', help='which model should be trained')
    parser.add_argument('--num_words', type=int, default=50000, help='size of the vocabulary')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')

    args = parser.parse_args()
    if args.model == 'word2vec':
        train_word2vec_curia(args.num_words, args.num_epochs)
    elif args.model == 'fasttext':
        train_fasttext_curia(args.num_words, args.num_epochs)
    elif args.model == 'elmo':
        train_elmo_curia()
