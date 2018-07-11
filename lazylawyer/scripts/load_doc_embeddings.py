import argparse
from lazylawyer.database import table_docs
from lazylawyer.content_generator import ContentGenerator
from lazylawyer import helpers
import os
import pickle

def load_doc_embeddings(file_name):
    """Loads document embeddings from a file and saves
    them in the database.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])

    with open(os.path.join('saved_embeddings', file_name), 'rb') as f:
        embs = pickle.load(f)

    for emb in embs:
        table_docs.update_embedding(emb['doc_id'], emb['emb'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save document embeddings using a certain model')
    parser.add_argument('file_name', help='file name of the pickled embeddings file')

    args = parser.parse_args()

    load_doc_embeddings(args.file_name)
