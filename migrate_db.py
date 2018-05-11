"""This script performs migration of the database.
"""
from database.database import CURIACaseDatabase

def main():
    db = CURIACaseDatabase()
    db.create_tables(remove_old=True)

if __name__ == '__main__':
    main()