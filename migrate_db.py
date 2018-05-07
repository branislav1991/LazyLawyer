"""This script performs migration of the database.
"""
from database.database import CURIACaseDatabase

db = CURIACaseDatabase()
db.create_tables(remove_old=True)