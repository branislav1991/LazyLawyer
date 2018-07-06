import argparse
from docai.database import table_docs, table_doc_contents
from flask import Flask, render_template, request
from docai import helpers
from docai.nlp.curia_preprocessor import preprocess
from docai.nlp import phrases
from docai.nlp.helpers import get_embedding_doc, cosine_similarity
import gensim
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

    query_emb = get_embedding_doc(search_query, model)

    similarities = [cosine_similarity(query_emb, pickle.loads(doc['embedding'])) for doc in docs]

    results = [{'link': doc['link'], 'name': doc['name'], 'abstract': abstract, 'similarity': sim} \
        for sim, doc, abstract in zip(similarities, docs, doc_abstracts)]
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return render_template('search_results.html', results=results[:50])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch flask web app.')
    parser.add_argument('model', choices=['word2vec', 'fasttext'], help='model for doc embeddings')
    parser.add_argument('model_path', help='model path')
    parser.add_argument('--num_words', type=int, default=100000, help='vocabulary size')

    args = parser.parse_args()

    if args.model == 'word2vec':
        print('Loading pretrained binary file...')
        model_path = os.path.join('trained_models', args.model_path)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=args.num_words)
        
    elif args.model == 'fasttext':
        print('Loading pretrained model...')
        model_path = os.path.join('trained_models', args.model_path)
        model = gensim.models.FastText.load(model_path)

    print('Loading documents...')
    docs = table_docs.get_docs_with_names(['Judgment'])
    doc_contents = [table_doc_contents.get_doc_content(doc) for doc in docs]
    doc_abstracts = [shorten(content, width=200) for content in doc_contents]

    app.run()
