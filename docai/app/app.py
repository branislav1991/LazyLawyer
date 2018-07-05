import argparse
from docai.database import table_docs, table_doc_contents
from flask import Flask, render_template, request
from docai import helpers
from docai.nlp.curia_preprocessor import preprocess
from docai.models.load_word2vec_binary import load_word2vec_binary
from docai.models.word2vec import Word2Vec, cosine_similarity
from docai.models.fasttext import FastText
from docai.models.vocabulary import Vocabulary, FastTextVocabulary
from docai.nlp import phrases
import os
import pickle
from textwrap import shorten

app = Flask(__name__)

averaging_scheme = 'average'

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_text']
    search_query = preprocess(search_query)

    query_emb = model.get_embedding_doc(search_query, strategy=averaging_scheme)

    similarities = [cosine_similarity(query_emb, pickle.loads(doc['embedding'])) for doc in docs]

    results = [{'link': doc['link'], 'name': doc['name'], 'abstract': abstract, 'similarity': sim} \
        for sim, doc, abstract in zip(similarities, docs, doc_abstracts)]
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return render_template('search_results.html', results=results[:50])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch flask web app.')
    parser.add_argument('--model', choices=['word2vec', 'word2vec_pretrained', 'fasttext'], default='word2vec', help='which model should be used')
    parser.add_argument('--avg_scheme', default='average', help='averaging scheme for queries; fasttext supports only average')
    parser.add_argument('--num_words', type=int, default=200000, help='size of the vocabulary for word2vec_pretrained model')

    args = parser.parse_args()
    averaging_scheme = args.avg_scheme

    if args.model == 'word2vec':
        model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
        meta_path = model_path + '_meta.pickle'

        print('Loading metainfo...')
        with open(meta_path, 'rb') as f:
            meta_info = pickle.load(f)

        vocabulary = Vocabulary(count=meta_info['word_count'])
        vocabulary.load_idf(meta_info['idf'])

        print('Loading model...')
        model = Word2Vec(vocabulary, meta_info['embedding_dim'])
        model.load(model_path)

    elif args.model == 'word2vec_pretrained':
        # load pretrained model from Google word2vec embeddings
        print('Loading pretrained binary file...')
        model_path = os.path.join('trained_models', helpers.setup_json['googlenews_word2vec_path'])
        word_dict = load_word2vec_binary(model_path, max_vectors=args.num_words)

        print('Creating vocabulary...')
        vocabulary = Vocabulary(vocab_dict=word_dict)
        idx_dict = vocabulary.load_from_dict(word_dict)

        print('Loading model...')
        model = Word2Vec(vocabulary, embedding_dim=300)
        model.load_from_dict(idx_dict)
        
    elif args.model == 'fasttext':
        model_path = os.path.join('trained_models', helpers.setup_json['fasttext_path'])
        meta_path = model_path + '_meta.pickle'

        print('Loading metainfo...')
        with open(meta_path, 'rb') as f:
            meta_info = pickle.load(f)

        vocabulary = FastTextVocabulary(count=meta_info['word_count'], max_ngram=meta_info['max_ngram'])

        print('Loading model...')
        model = FastText(vocabulary, meta_info['embedding_dim'])
        model.load(model_path)

    print('Loading documents...')
    docs = table_docs.get_docs_with_names(['Judgment'])
    doc_contents = [table_doc_contents.get_doc_content(doc) for doc in docs]
    doc_abstracts = [shorten(content, width=200) for content in doc_contents]

    app.run()
