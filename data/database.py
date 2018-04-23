import sqlite3

class CaseDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

    def _create_table(self, name,  remove_old=False):
        """Creates a table in the database.
        Input params:
        remove_old: if table with the given names already exists
        and remove_old=True, we remove it and replace it with
        a new one.
        """
        pass

class CURIACaseDatabase(CaseDatabase):
    def __init__(self, db_path):
        super().__init__(db_path)
    
    def create_tables(self, remove_old=False):
        """Creates tables in the database.
        Input params:
        remove_old: if tables with the given names already exist
        and remove_old=True, we remove them and replace them with
        new ones.
        """
        if remove_old:
            sql = "DROP TABLE IF EXISTS cases;"
            self.cur.execute(sql)
            sql = "DROP TABLE IF EXISTS docs;"
            self.cur.execute(sql)

        sql = """CREATE TABLE IF NOT EXISTS cases (
                id integer PRIMARY KEY,
                name text NOT NULL,
                desc text NOT NULL,
                url text NOT NULL
            );"""
        self.cur.execute(sql)
        sql = """CREATE TABLE IF NOT EXISTS docs (
                id integer PRIMARY KEY,
                case_id integer,
                name text NOT NULL,
                ecli text,
                date text NOT NULL,
                parties text NOT NULL,
                subject text NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(id)
            );"""
        self.cur.execute(sql)
    
    def write_cases(self, cases):
        """Stores cases to database.
        Input params:
        cases: case dictionary with all relevant information.
        """
        def insert_case(case):
            sql = "INSERT INTO cases VALUES (?,?,?)"
            self.cur.execute(sql, (case['name'], case['desc'], case['url']))
        map(insert_case, cases)
    
    def write_docs(self, case, docs):
        """Stores documents belonging to a case.
        Input params:
        case: the case that the documents belong to.
        docs: a dictionary of documents.
        """
        sql = "SELECT id FROM cases WHERE name=? AND desc=? AND url=?"
        self.cur.execute(sql, (case['name'], case['desc'], case['url']))
        case_id = self.cur.fetchall()

        def insert_doc(doc):
            sql = "INSERT INTO docs VALUES (?,?,?,?,?,?)"
            self.cur.execute(sql, (case_id, doc['name'], doc['ecli'], doc['date'], doc['parties'], doc['subject']))
        map(insert_doc, docs)