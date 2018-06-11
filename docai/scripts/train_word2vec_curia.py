from docai.database import table_docs, table_doc_contents
from docai.content_generator import ContentGenerator
from docai import helpers
from itertools import chain
from docai.nlp.curia_preprocessor import preprocess
from docai.models.word2vec import Word2Vec
from docai.nlp.vocabulary import Vocabulary
from docai.nlp import phrases
import os

def train_word2vec_curia():
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_name('Judgment')
    content_gen = ContentGenerator(docs)

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
    vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

    contents = list(content_gen) # generate all contents at once

    print('Initializing vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.initialize_and_save_vocab(contents, vocab_path)
    print('Initializing idf weights...')
    vocabulary.initialize_and_save_idf(contents, vocab_path)

    print('Initializing model...')
    model = Word2Vec(vocabulary)
    print('Starting training...')
    model.train(contents, model_path)

if __name__ == '__main__':
    train_word2vec_curia()
