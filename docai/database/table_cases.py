from docai.database import database as db

def write_cases(cases):
    """Stores cases to database.
    Input params:
    cases: case dictionary with all relevant information.
    """
    db.batch_insert_check('cases', cases, attrs=['name'])

def get_all_cases():
    """Retrieves all cases from the database.
    This can be useful e.g. when we want to
    bulk download or crawl.
    """
    s = """SELECT * FROM cases"""
    db.cursor.execute(s)
    rows = db.cursor.fetchall()
    if not rows:
        return None
    else:
        return db._convert_to_cases_dict(rows)

def get_case_with_name(name):
    """Retrieves case with a given name. Returns
    None if none found.
    """
    s = """SELECT * FROM cases WHERE name=?"""
    db.cursor.execute(s, (name,))
    rows = db.cursor.fetchone()
    if not rows:
        return None
    else:
        return db._convert_to_cases_dict([rows])[0]

def get_case_for_doc(doc):
    """Retrieves case for a document.
    """
    s = """SELECT * FROM cases WHERE id=?"""
    db.cursor.execute(s, (doc['case_id'],))
    rows = db.cursor.fetchone()
    if not rows:
        return None
    else:
        return db._convert_to_cases_dict([rows])[0]

def update_subject(case, subject):
    """Updates the subject of the case.
    Subject is a text field.
    """
    s = """UPDATE cases SET subject=? WHERE id=?"""
    result = db.cursor.execute(s, (subject, case['id']))
    db.connection.commit()

def update_parties(case, party1, party2):
    """Updates the category of the case.
    Category is a text field.
    """
    if party1 is not None:
        s = """UPDATE cases SET party1=? WHERE id=?"""
        result = db.cursor.execute(s, (party1, case['id']))
        db.connection.commit()

    if party2 is not None:
        s = """UPDATE cases SET party2=? WHERE id=?"""
        result = db.cursor.execute(s, (party2, case['id']))
        db.connection.commit()