from database.database import CURIACaseDatabase
from nlp import doc_similarity

db = CURIACaseDatabase()

docs = db.get_docs_with_name('Judgment')
docs = docs[:10]
contents = [db.get_doc_content(doc).decode() for doc in docs]

for content1 in contents:
    for content2 in contents:
        sim = doc_similarity.similarity(content1, content2)
        print('Similarity of document 1 and document 2 is {0}'.format(sim))

