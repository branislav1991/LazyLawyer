from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.models.fasttext import FastText
from docai.nlp.vocabulary import FastTextVocabulary
import os

def train_fasttext_curia():
    print("Initializing database and loading documents...")
    docs = table_docs.get_docs_with_name('Judgment')
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
    model = FastText(vocabulary, epoch_num=10)
    print('Starting training...')
    model.train(contents, model_path)

    print('Saving document embeddings...')
    save_doc_embeddings(vocabulary, model)

def save_doc_embeddings(vocabulary, model):
    docs = table_docs.get_docs_with_name('Judgment')
    content_gen = ContentGenerator(docs)

    for doc, content in zip(docs, content_gen):
        emb = model.get_embedding_doc(content, strategy='tf-idf')
        table_docs.update_embedding(doc, emb)

if __name__ == '__main__':
    train_fasttext_curia()
