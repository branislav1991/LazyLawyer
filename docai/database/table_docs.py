from docai.database import database as db
import pickle
import sqlite3

def write_docs_for_case(case, docs):
    """Stores documents belonging to a case.
    Input params:
    case: the case that the documents belong to.
    docs: a dictionary of documents.
    """
    s = """SELECT id FROM cases WHERE name=? AND desc=? AND url=?"""
    db.cursor.execute(s, (case['name'], case['desc'], case['url']))
    row = db.cursor.fetchone()

    for doc in docs:
        doc['case_id'] = row[0]
    db.batch_insert_check('docs', docs, attrs=['case_id', 'name'])

def get_max_case_id_in_docs():
    """Retrieves the highest case_id present in the
    docs table. This is to allow for appending the docs
    table with new cases.
    """
    s = """SELECT MAX(case_id) FROM docs"""
    db.cursor.execute(s)
    row = db.cursor.fetchone()
    return -1 if row[0] is None else row[0]

def get_docs_for_case(case, only_with_link=True, downloaded=True):
    """Retrieves documents for a specific case.
    Input params:
    case: case to get docs for.
    only_with_link: only retrieve docs which contain a link.
    downloaded: if True, also retrieves documents which are already
    downloaded (successfully or unsuccessfully)
    """
    s = """SELECT * FROM docs WHERE case_id=?"""
    if only_with_link:
        s += """ AND link IS NOT NULL"""
    if not downloaded:
        s += """ AND download_error IS NULL"""

    db.cursor.execute(s, (case['id'],))
    rows = db.cursor.fetchall()
    return db._convert_to_docs_dict(rows)

def get_docs_with_name(name, only_valid=True):
    """Retrieves all documents with a specific name.
    Input params:
    name: name of the document to retrieve.
    only_valid: only retrieve docs which contain a link.
    """
    if only_valid:
        s = """SELECT * FROM docs WHERE name=? AND link IS NOT NULL"""
    else:
        s = """SELECT * FROM docs WHERE name=?"""

    db.cursor.execute(s, (name,))
    rows = db.cursor.fetchall()
    return db._convert_to_docs_dict(rows)

def get_doc_case(doc):
    """Retrieves case for a document.
    """
    s = """SELECT * FROM cases WHERE id=?"""
    db.cursor.execute(s, (doc['case_id'],))
    rows = db.cursor.fetchone()
    return db._convert_to_cases_dict([rows])[0]

def write_download_error(doc, result):
    s = """UPDATE docs SET download_error=? WHERE id=?"""
    result = db.cursor.execute(s, (result, doc['id']))
    db.connection.commit()

def update_embedding(doc, embedding):
    pdata = pickle.dumps(embedding, pickle.HIGHEST_PROTOCOL)
    s = """UPDATE docs SET embedding=? WHERE id=?"""
    result = db.cursor.execute(s, (sqlite3.Binary(pdata), doc['id']))
    db.connection.commit()
