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

    def _convert_to_cases_dict(self, db_rows):
        cases = [{'id': x[0], 'name': x[1], 'desc': x[2], 'url': x[3]} for x in db_rows]
        return cases
    
    def _convert_to_docs_dict(self, db_rows):
        docs = [{'id': x[0], 'case_id': x[1], 'name': x[2], 'ecli': x[3], 'date': x[4],
                'parties': x[5], 'subject': x[6], 'link_curia': x[7],
                'link_eurlex': x[8], 'type': x[9]}]

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
            type TEXT,
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

    def get_all_cases(self):
        """Retrieves all cases from the database.
        This can be useful e.g. when we want to
        bulk download or crawl.
        """
        s = """SELECT * FROM cases"""
        result = self.cursor.execute(s)
        rows = result.fetchall()
        return self._convert_to_cases_dict(rows)

    def get_max_case_id_in_docs(self):
        """Retrieves the highest case_id present in the
        docs table. This is to allow for appending the docs
        table with new cases.
        """
        s = """SELECT MAX(case_id) FROM docs"""
        result = self.cursor.execute(s)
        row = result.fetchone()
        return row[0]

    def get_docs_for_case(self, case):
        """Retrieves documents for a specific case.
        """
        
        s = """SELECT * FROM docs WHERE case_id=?"""
        result = self.cursor.execute(s, (case['id']))
        rows = result.fetchall()
        return self._convert_to_docs_dict(rows)