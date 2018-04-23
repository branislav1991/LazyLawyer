import sqlite3

class CaseDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)