import helpers
import json
import sqlite3

class CaseDatabase:
    SETUP_FILE_PATH = 'database/db_setup.json'

    def __init__(self):
        with open(CaseDatabase.SETUP_FILE_PATH, 'r') as setup_file:
            setup_json = json.load(setup_file) 
            db_path = setup_json['db_path']
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

    def close(self):
        self.connection.close()

    def _batch_insert(self, table, vals, batch_size=100):
        batches = list(helpers.create_batches(vals, batch_size))
        for batch in batches:
            cols = batch[0].keys()
            batchvals = [tuple(x.values()) for x in batch]
            placeholder = '?'
            placeholderlist = ",".join([placeholder] * len(cols))

            s = 'INSERT INTO '
            s += table
            s += '(' + ','.join(cols) + ')'
            s += ' VALUES'
            s += '(' + placeholderlist + ')'
            self.cursor.executemany(s, batchvals)
            self.connection.commit()

class CURIACaseDatabase(CaseDatabase):
    def __init__(self):
        super().__init__()

    def create_tables(self, remove_old=False):
        if remove_old:
            self.cursor.execute("""DROP TABLE IF EXISTS cases""")
            self.cursor.execute("""DROP TABLE IF EXISTS docs""")

        self.cursor.execute("""CREATE TABLE IF NOT EXISTS cases(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            desc TEXT NOT NULL, 
            url TEXT NOT NULL 
            )""") 
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            name TEXT,
            ecli TEXT,
            date TEXT, 
            parties TEXT,
            subject TEXT,
            link_curia TEXT,
            link_eurlex TEXT,
            FOREIGN KEY (case_id) REFERENCES cases(id) 
            )""")

        self.connection.commit()

    def write_cases(self, cases):
        """Stores cases to database.
        Input params:
        cases: case dictionary with all relevant information.
        """
        self._batch_insert('cases', cases)
    
    def write_docs(self, case, docs):
        """Stores documents belonging to a case.
        Input params:
        case: the case that the documents belong to.
        docs: a dictionary of documents.
        """
        s = """SELECT id FROM cases WHERE name=? AND desc=? AND url=?"""
        result = self.cursor.execute(s, (case['name'], case['desc'], case['url']))
        row = result.fetchone()

        for doc in docs:
            doc['case_id'] = row[0]
        self._batch_insert('docs', docs)