from database import table_docs, table_doc_contents
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
    docs = docs[:10]
    doc_abstracts = [table_doc_contents.get_doc_content(x) for x in docs]
    doc_abstracts = [shorten(x, width=200) for x in doc_abstracts]
    results = [{'name': doc['name'], 'abstract': abstract} for doc, abstract in zip(docs, doc_abstracts)]

    return render_template('search_results.html', results=results)