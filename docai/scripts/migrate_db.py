"""This script performs migration of the database.
"""
from docai.database import database as db

def migrate_db():
    db.create_tables(remove_old=True)

if __name__ == '__main__':
    migrate_db()
