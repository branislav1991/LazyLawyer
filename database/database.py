"""Database interface. The database is made
thread-safe by locking the cursor. Therefore it
is safe to call database functions from multiple
threads at once.
"""
import atexit
import helpers
import json
import sqlite3
import threading

SETUP_FILE_PATH = 'database/db_setup.json'

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

    def lastrowid(self):
        return self._cursor.lastrowid

# initialize database connection
with open(SETUP_FILE_PATH, 'r') as setup_file:
    setup_json = json.load(setup_file) 
    db_path = setup_json['db_path']
connection = sqlite3.connect(db_path, check_same_thread=False)
cursor = ThreadSafeCursor(connection)

def close():
    """Closes database connection. This method gets
    called automatically at program exit.
    """
    connection.close()
atexit.register(close)

def _get_non_existing_entries(batch, table, attrs):
    """For a batch of values, check if those
    values already exist in the table based on 
    a specific attribute. Returns all entries
    that are non-existent in the database.
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
    cursor.execute(s)
    rows = cursor.fetchall()

    batchnames = [x[0] for x in rows]
    batch = [x for x in batch if x['name'] not in batchnames]
    return batch

def _insert_batch(batch, table):
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
    cursor.executemany(s, batchvals)
    connection.commit()

def batch_insert_check(table, vals, attrs, batch_size=100):
    """Process vals in batches of a specific batch size
    and insert them into a table. Also perform a check to
    verify if data is already in the database based on the attributes
    (columns) defined in attrs.
    """
    batches = list(helpers.create_batches_generate(vals, batch_size))
    for batch in batches:
        batch = _get_non_existing_entries(batch, table, attrs)
        if len(batch) > 0:
            _insert_batch(batch, table)

def _convert_to_cases_dict(db_rows):
    cases = [{'id': x[0], 'name': x[1], 'desc': x[2],
        'url': x[3], 'protocol': x[4]} for x in db_rows]
    return cases

def _convert_to_docs_dict(db_rows):
    docs = [{'id': x[0], 'case_id': x[1], 'name': x[2], 'ecli': x[3], 'date': x[4],
            'parties': x[5], 'subject': x[6], 'link': x[7],
            'source': x[8], 'format': x[9], 'content_id': x[10],
            'download_error': x[11]} for x in db_rows]
    return docs

def create_tables(remove_old=False):
    if remove_old:
        cursor.execute("""DROP TABLE IF EXISTS cases""")
        cursor.execute("""DROP TABLE IF EXISTS docs""")

    # download_error column indicates if there was an error downloading docs for 
    # the case. If 0, no problem, of 1, problem. If NULL, no download attempts yet.
    cursor.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        desc TEXT NOT NULL, 
        url TEXT NOT NULL,
        protocol TEXT NOT NULL
        )""") 
    cursor.execute("""CREATE TABLE IF NOT EXISTS docs(
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
    cursor.execute("""CREATE TABLE IF NOT EXISTS doc_contents(
        id INTEGER PRIMARY KEY,
        content BLOB NOT NULL,
        doc_id INTEGER NOT NULL,
        FOREIGN KEY (doc_id) REFERENCES docs(id)
        )""")

    connection.commit()