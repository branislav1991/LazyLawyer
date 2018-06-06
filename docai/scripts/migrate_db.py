"""This script performs migration of the database.
"""
from docai.database import database as db

def main():
    db.create_tables(remove_old=True)

if __name__ == '__main__':
    main()
