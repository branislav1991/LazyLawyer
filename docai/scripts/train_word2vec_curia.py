import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.models.word2vec import Word2Vec
from docai.nlp.vocabulary import Vocabulary
import os

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

def save_doc_embeddings(vocabulary, model):
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    for doc, content in zip(docs, content_gen):
        emb = model.get_embedding_doc(content, strategy='average')
        table_docs.update_embedding(doc, emb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run whole crawling pipeline up to document content saving')
    parser.add_argument('--num_words', type=int, default=50000, help='size of the vocabulary')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')

    args = parser.parse_args()
    train_word2vec_curia(args.num_words, args.num_epochs)
