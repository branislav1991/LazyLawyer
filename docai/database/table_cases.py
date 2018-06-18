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
    return db._convert_to_cases_dict(rows)

def get_case_for_doc(doc):
    """Retrieves case for a document.
    """
    s = """SELECT * FROM cases WHERE id=?"""
    db.cursor.execute(s, (doc['case_id'],))
    rows = db.cursor.fetchone()
    return db._convert_to_cases_dict([rows])[0]

def update_category(case, category):
    """Updates the category of the case.
    Category is a text field.
    """
    s = """UPDATE cases SET category=? WHERE id=?"""
    result = db.cursor.execute(s, (category, case['id']))
    db.connection.commit()
