"""Database interface. The database is made
thread-safe by locking the cursor. Therefore it
is safe to call database functions from multiple
threads at once.
"""

import helpers
import json
import sqlite3
import threading

class ThreadSafeCursor:
    _lock = threading.Lock()

    def __init__(self, connection):
        self._cursor = connection.cursor()

    def execute(self, str, *params):
        with ThreadSafeCursor._lock:
            result = self._cursor.execute(str, *params)
        return result

    def executemany(self, str, seq_of_params):
        with ThreadSafeCursor._lock:
            result = self._cursor.executemany(str, seq_of_params)
        return result

    def fetchone(self):
        with ThreadSafeCursor._lock:
            result = self._cursor.fetchone()
        return result

    def fetchall(self):
        with ThreadSafeCursor._lock:
            result = self._cursor.fetchall()
        return result

class CaseDatabase:
    SETUP_FILE_PATH = 'database/db_setup.json'

    def __init__(self):
        with open(CaseDatabase.SETUP_FILE_PATH, 'r') as setup_file:
            setup_json = json.load(setup_file) 
            db_path = setup_json['db_path']
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = ThreadSafeCursor(self.connection)

    def close(self):
        self.connection.close()

    def _batch_check_existing(self, batch, table, attrs):
        """For a batch of values, check if those
        values already exist in the table based on 
        a specific attribute.
        """
        conditions = []
        for attr in attrs:
            batchvals = [x[attr] for x in batch]
            cond = attr + ' IN ('
            cond +=  ','.join('"{0}"'.format(b) for b in batchvals)
            cond += ')'
            conditions.append(cond)

        s = 'SELECT '
        s += attr
        s += ' FROM '
        s += table
        s += ' WHERE '
        s += ' AND '.join(conditions)
        self.cursor.execute(s)
        rows = self.cursor.fetchall()

        batchnames = [x[0] for x in rows]
        batch = [x for x in batch if x['name'] not in batchnames]
        return batch

    def _insert_batch(self, batch, table):
        """Insert one batch of data into a table.
        """
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

    def batch_insert_check(self, table, vals, attrs, batch_size=100):
        """Process vals in batches of a specific batch size
        and insert them into a table. Also perform a check to
        verify if data is already in the database based on the attributes
        (columns) defined in attrs.
        """
        batches = list(helpers.create_batches_generate(vals, batch_size))
        for batch in batches:
            batch = self._batch_check_existing(batch, table, attrs)
            if len(batch) > 0:
                self._insert_batch(batch, table)

class CURIACaseDatabase(CaseDatabase):
    def __init__(self):
        super().__init__()

    def _convert_to_cases_dict(self, db_rows):
        cases = [{'id': x[0], 'name': x[1], 'desc': x[2],
            'url': x[3], 'protocol': x[4]} for x in db_rows]
        return cases
    
    def _convert_to_docs_dict(self, db_rows):
        docs = [{'id': x[0], 'case_id': x[1], 'name': x[2], 'ecli': x[3], 'date': x[4],
                'parties': x[5], 'subject': x[6], 'link': x[7],
                'source': x[8], 'format': x[9], 'content_id': x[10],
                'download_error': x[11]} for x in db_rows]
        return docs

    def create_tables(self, remove_old=False):
        if remove_old:
            self.cursor.execute("""DROP TABLE IF EXISTS cases""")
            self.cursor.execute("""DROP TABLE IF EXISTS docs""")

        # download_error column indicates if there was an error downloading docs for 
        # the case. If 0, no problem, of 1, problem. If NULL, no download attempts yet.
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS cases(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            desc TEXT NOT NULL, 
            url TEXT NOT NULL,
            protocol TEXT NOT NULL
            )""") 
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            name TEXT,
            ecli TEXT,
            date TEXT, 
            parties TEXT,
            subject TEXT,
            link TEXT,
            source TEXT,
            format TEXT,
            content_id INTEGER,
            download_error INTEGER,
            FOREIGN KEY (case_id) REFERENCES cases(id),
            FOREIGN KEY (content_id) REFERENCES doc_contents(id)
            )""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS doc_contents(
            id INTEGER PRIMARY KEY,
            content BLOB NOT NULL,
            doc_id INTEGER NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES docs(id)
            )""")

        self.connection.commit()

    def write_cases(self, cases):
        """Stores cases to database.
        Input params:
        cases: case dictionary with all relevant information.
        """
        self.batch_insert_check('cases', cases, attrs=['name'])
    
    def write_docs(self, case, docs):
        """Stores documents belonging to a case.
        Input params:
        case: the case that the documents belong to.
        docs: a dictionary of documents.
        """
        s = """SELECT id FROM cases WHERE name=? AND desc=? AND url=?"""
        self.cursor.execute(s, (case['name'], case['desc'], case['url']))
        row = self.cursor.fetchone()

        for doc in docs:
            doc['case_id'] = row[0]
        self.batch_insert_check('docs', docs, attrs=['case_id', 'name'])

    def write_doc_content(self, doc, text):
        """Stores text for a document. Requires
        doc['id'] to be stored in the doc dict.
        """
        if doc['content_id'] is not None: # check if no content assigned yet
            return

        s = """INSERT INTO doc_contents (content, doc_id) VALUES (?, ?)"""
        self.cursor.execute(s, (text.encode(), doc['id']))
        self.connection.commit()

        self.cursor.lastrowid
        s = """UPDATE docs SET content_id=? WHERE id=?"""
        self.cursor.execute(s, (self.cursor.lastrowid, doc['id']))
        self.connection.commit()

    def get_all_cases(self):
        """Retrieves all cases from the database.
        This can be useful e.g. when we want to
        bulk download or crawl.
        """
        s = """SELECT * FROM cases"""
        self.cursor.execute(s)
        rows = self.cursor.fetchall()
        return self._convert_to_cases_dict(rows)

    def get_max_case_id_in_docs(self):
        """Retrieves the highest case_id present in the
        docs table. This is to allow for appending the docs
        table with new cases.
        """
        s = """SELECT MAX(case_id) FROM docs"""
        self.cursor.execute(s)
        row = self.cursor.fetchone()
        return -1 if row[0] is None else row[0]

    def get_docs_for_case(self, case, only_with_link=True, downloaded=True):
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

        self.cursor.execute(s, (case['id'],))
        rows = self.cursor.fetchall()
        return self._convert_to_docs_dict(rows)

    def get_docs_with_name(self, name, only_valid=True):
        """Retrieves all documents with a specific name.
        Input params:
        name: name of the document to retrieve.
        only_valid: only retrieve docs which contain a link.
        """
        if only_valid:
            s = """SELECT * FROM docs WHERE name=? AND link IS NOT NULL"""
        else:
            s = """SELECT * FROM docs WHERE name=?"""

        self.cursor.execute(s, (name,))
        rows = self.cursor.fetchall()
        return self._convert_to_docs_dict(rows)

    def get_doc_case(self, doc):
        """Retrieves case for a document.
        """
        s = """SELECT * FROM cases WHERE id=?"""
        self.cursor.execute(s, (doc['case_id'],))
        rows = self.cursor.fetchone()
        return self._convert_to_cases_dict([rows])[0]

    def write_download_error(self, case, result):
        s = """UPDATE docs SET download_error=? WHERE case_id=?"""
        result = self.cursor.execute(s, (result, case['id']))
        self.connection.commit()