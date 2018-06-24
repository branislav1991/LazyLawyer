import argparse
from docai.database import table_docs
from docai.content_generator import ContentGenerator
from docai import helpers
from docai.models.word2vec import Word2Vec
from docai.models.load_word2vec_binary import load_word2vec_binary
from docai.nlp.vocabulary import Vocabulary
import os

def save_word2vec_pretrained_curia(num_words):
    # load pretrained model from Google word2vec embeddings
    print('Loading pretrained binary file...')
    model_path = os.path.join('trained_models', helpers.setup_json['googlenews_word2vec_path'])
    word_dict = load_word2vec_binary(model_path, max_vectors=num_words)

    print('Creating vocabulary...')
    vocabulary = Vocabulary()
    idx_dict = vocabulary.load_from_dict(word_dict)

    print('Loading model...')
    model = Word2Vec(vocabulary, embedding_dim=300)
    model.load_from_dict(idx_dict)

    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    for doc, content in zip(docs, content_gen):
        emb = model.get_embedding_doc(content, strategy='average')
        table_docs.update_embedding(doc, emb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run whole crawling pipeline up to document content saving')
    parser.add_argument('--num_words', type=int, default=200000, help='size of the vocabulary')

    args = parser.parse_args()
    save_word2vec_pretrained_curia(args.num_words)
