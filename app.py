from database import table_docs, table_doc_contents
from nlp import doc_similarity
from flask import Flask, render_template, request
from textwrap import shorten

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_text']

    docs = table_docs.get_docs_with_name('Judgment')
    docs = docs[:100] # for testing, take first 100 documents
    
    doc_contents = [table_doc_contents.get_doc_content(x) for x in docs]

    similarities = [doc_similarity.similarity(search_query, content) for content in doc_contents]

    doc_abstracts = [shorten(x, width=200) for x in doc_contents]

    results = [{'link': doc['link'], 'name': doc['name'], 'abstract': abstract, 'similarity': sim} \
        for sim, doc, abstract in zip(similarities, docs, doc_abstracts)]
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return render_template('search_results.html', results=results)