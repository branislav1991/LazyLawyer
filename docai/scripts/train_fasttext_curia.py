import argparse
from docai import helpers
from docai.models.fasttext import FastText
from docai.nlp.vocabulary import FastTextVocabulary
from docai.save_doc_embeddings import save_doc_embeddings
import os

def train_fasttext_curia(num_ngrams, epoch_num):
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['fasttext_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    contents = list(content_gen) # generate all contents at once

    print('Initializing vocabulary...')
    vocabulary = FastTextVocabulary()
    vocabulary.initialize_and_save_vocab(contents, vocab_path)
    print('Initializing idf weights...')
    vocabulary.initialize_and_save_idf(contents, vocab_path)

    print('Initializing model...')
    model = FastText(vocabulary)
    print('Starting training...')
    model.train(contents, model_path, epoch_num=epoch_num)

    print('Saving document embeddings...')
    save_doc_embeddings(vocabulary, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run whole crawling pipeline up to document content saving')
    parser.add_argument('--num_ngrams', type=int, default=400000, help='size of the n-gram vocabulary')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')

    args = parser.parse_args()
    train_fasttext_curia(args.num_ngrams, args.num_epochs)
