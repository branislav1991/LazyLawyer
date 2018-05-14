from database import table_docs, table_doc_contents
from nlp import doc_similarity

docs = table_docs.get_docs_with_name('Judgment')
docs = docs[:5]
contents = [table_doc_contents.get_doc_content(doc) for doc in docs]

for content1 in contents:
    for content2 in contents:
        sim = doc_similarity.similarity(content1, content2)
        print('Similarity of document 1 and document 2 is {0}'.format(sim))

