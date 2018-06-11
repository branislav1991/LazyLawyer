from docai.database import table_docs, table_doc_contents
from flask import Flask, render_template, request
from docai import helpers
from docai.nlp.curia_preprocessor import preprocess
from docai.models.word2vec import Word2Vec, cosine_similarity
from docai.nlp.vocabulary import Vocabulary
from docai.nlp import phrases
import os
import pickle
from textwrap import shorten

app = Flask(__name__)

model_path = os.path.join('trained_models', helpers.setup_json['word2vec_path'])
vocab_path = os.path.join('trained_models', helpers.setup_json['vocab_path'])

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_text']
    search_query = preprocess(search_query)
    search_query = phrases.build_phrases_regex(search_query)

    query_emb = model.get_embedding_doc(search_query, strategy='tf-idf')

    similarities = [cosine_similarity(query_emb, pickle.loads(doc['embedding'])) for doc in docs]# if doc['vector'] is not None]

    results = [{'link': doc['link'], 'name': doc['name'], 'abstract': abstract, 'similarity': sim} \
        for sim, doc, abstract in zip(similarities, docs, doc_abstracts)]
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return render_template('search_results.html', results=results[:50])

if __name__ == '__main__':
    print('Loading vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.load(vocab_path)

    print('Loading model...')
    model = Word2Vec(vocabulary)
    model.load(model_path)

    print('Loading documents...')
    docs = table_docs.get_docs_with_name('Judgment')
    doc_contents = [table_doc_contents.get_doc_content(doc) for doc in docs]
    doc_abstracts = [shorten(content, width=200) for content in doc_contents]

    app.run()
