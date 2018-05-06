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
        result = self.cursor.execute(s)
        rows = result.fetchall()

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
        batches = list(helpers.create_batches(vals, batch_size))
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
                'source': x[8], 'format': x[9]} for x in db_rows]
        return docs

    def create_tables(self, remove_old=False):
        if remove_old:
            self.cursor.execute("""DROP TABLE IF EXISTS cases""")
            self.cursor.execute("""DROP TABLE IF EXISTS docs""")

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
        result = self.cursor.execute(s, (case['name'], case['desc'], case['url']))
        row = result.fetchone()

        for doc in docs:
            doc['case_id'] = row[0]
        self.batch_insert_check('docs', docs, attrs=['case_id', 'name'])

    def get_doc_content(self, doc):
        """Retrieves text for a document.
        """
        s = """SELECT content FROM doc_contents WHERE doc_id=?"""
        result = self.cursor.execute(s, (doc['id'],))
        row = result.fetchone()
        if row is None:
            text = None
        else:
            text = row[0].decode()

        return text

    def write_doc_content(self, doc, text):
        """Stores text for a document. Requires
        doc['id'] to be stored in the doc dict.
        Also checks if doc_content is already stored for this doc.
        """
        if self.get_doc_content(doc) is not None:
            return

        s = """INSERT INTO doc_contents (content, doc_id) VALUES (?, ?)"""
        self.cursor.execute(s, (text.encode(), doc['id']))
        self.connection.commit()

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
        return -1 if row[0] is None else row[0]

    def get_docs_for_case(self, case, only_valid=True):
        """Retrieves documents for a specific case.
        Input params:
        case: case to get docs for.
        only_valid: only retrieve docs which contain a link.
        """
        if only_valid:
            s = """SELECT * FROM docs WHERE case_id=? AND link IS NOT NULL"""
        else:
            s = """SELECT * FROM docs WHERE case_id=?"""

        result = self.cursor.execute(s, (case['id'],))
        rows = result.fetchall()
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

        result = self.cursor.execute(s, (name,))
        rows = result.fetchall()
        return self._convert_to_docs_dict(rows)

    def get_doc_case(self, doc):
        """Retrieves case for a document.
        """
        s = """SELECT * FROM cases WHERE id=?"""
        result = self.cursor.execute(s, (doc['case_id'],))
        rows = result.fetchone()
        return self._convert_to_cases_dict([rows])[0]