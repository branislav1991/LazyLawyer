"""Calculates embeddings for all documents in the database
and stores the results in the 'embedding' column of the docs table.
"""
from docai.database import table_docs, table_doc_contents
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.models.word2vec import Word2Vec
from docai.nlp.curia_preprocessor import preprocess
from docai.nlp.vocabulary import Vocabulary
from docai.nlp import phrases
import os

model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

def main():
    print('Loading vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.load(vocab_path)

    print('Loading model...')
    model = Word2Vec(vocabulary)
    model.load(model_path)

    docs = table_docs.get_docs_with_name('Judgment')
    content_gen = ContentGenerator(docs)

    for doc, content in zip(docs, content_gen):
        emb = model.get_embedding_doc(content, strategy='tf-idf')
        table_docs.update_embedding(doc, emb)

if __name__ == '__main__':
    main()
