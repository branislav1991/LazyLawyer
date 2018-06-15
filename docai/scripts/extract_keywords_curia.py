from docai.nlp.curia_preprocessor import extract_keywords
from docai.database import table_docs, table_doc_contents

def extract_keywords_curia():
    docs = table_docs.get_docs_with_name('Judgment')
    contents = (table_doc_contents.get_doc_content(doc) for doc in docs)
    for doc, content in zip(docs, contents):
        keywords = extract_keywords(content)
        if keywords:
            table_docs.update_keywords(doc, extract_keywords(content))

if __name__ == '__main__':
    extract_keywords_curia()
