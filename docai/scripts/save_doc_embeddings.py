"""Calculates embeddings for all documents in the database
and stores the results in the 'embedding' column of the docs table.
"""
from docai.database import table_docs, table_doc_contents
from docai import helpers
from docai.nlp.word2vec_model import Word2Vec
from docai.nlp.curia_preprocessor import preprocess
from docai.nlp.vocabulary import Vocabulary
from docai.nlp import phrases
import os

save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

def main():
    print('Loading vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.load(save_path)

    print('Loading model...')
    model = Word2Vec(vocabulary)
    model.load(save_path)

    docs = table_docs.get_docs_with_name('Judgment')

    for doc in docs:
        content = preprocess(table_doc_contents.get_doc_content(doc))
        content = phrases.build_phrases_regex(content)
        emb = model.get_embedding_doc(content, strategy='tf-idf')
        table_docs.update_embedding(doc, emb)

if __name__ == '__main__':
    main()
