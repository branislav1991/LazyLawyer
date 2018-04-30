"""Performs migration of the database.
"""
from database import CURIACaseDatabase

db = CURIACaseDatabase()
db.create_tables(remove_old=True)