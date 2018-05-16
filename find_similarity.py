from database import table_docs, table_doc_contents
from nlp import doc_similarity

most_similar = doc_similarity.most_similar('court', 5)
print(most_similar)
