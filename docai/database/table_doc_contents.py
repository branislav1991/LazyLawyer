from docai.database import database as db

def write_doc_content(doc, text):
    """Stores text for a document. Requires
    doc['id'] to be stored in the doc dict.
    """
    if doc['content_id'] is not None: # check if no content assigned yet
        return

    s = """INSERT INTO doc_contents (content, doc_id) VALUES (?, ?)"""
    db.cursor.execute(s, (text.encode(), doc['id']))
    db.connection.commit()

    s = """UPDATE docs SET content_id=? WHERE id=?"""
    db.cursor.execute(s, (db.cursor.lastrowid(), doc['id']))
    db.connection.commit()

def get_doc_content(doc):
    """Returns content for a document or None if
    no content was stored. Requires doc['id'] to be
    stored in the doc dict.
    """
    if doc['content_id'] is None:
        return None
    
    s = """SELECT content FROM doc_contents WHERE id=?"""
    db.cursor.execute(s, (doc['content_id'],))
    row = db.cursor.fetchone()
    return row[0].decode()
