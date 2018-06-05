from database import table_docs, table_doc_contents
from flask import Flask, render_template, request
import helpers
from nlp.curia_preprocessor import preprocess
from nlp.word2vec_model import Word2Vec
from nlp.vocabulary import Vocabulary
from nlp import phrases
import os
from textwrap import shorten

app = Flask(__name__)

save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

print('Loading vocabulary...')
vocabulary = Vocabulary()
vocabulary.load(save_path)

print('Loading model...')
model = Word2Vec(vocabulary)
model.load(save_path)

print('Loading documents...')
docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:100]
doc_contents = [table_doc_contents.get_doc_content(doc) for doc in docs]
doc_abstracts = [shorten(content, width=200) for content in doc_contents]
doc_contents = [phrases.build_phrases_regex(preprocess(content)) for content in doc_contents]

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_text']
    search_query = preprocess(search_query)
    search_query = phrases.build_phrases_regex(search_query)

    similarities = [model.doc_similarity(search_query, content, strategy='tf-idf') for content in doc_contents]# if doc['vector'] is not None]

    results = [{'link': doc['link'], 'name': doc['name'], 'abstract': abstract, 'similarity': sim} \
        for sim, doc, abstract in zip(similarities, docs, doc_abstracts)]
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return render_template('search_results.html', results=results)
