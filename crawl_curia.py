"""This script crawls the CURIA database and saves all cases
and relevant pdf document links for each case to the database.
"""

from crawlers.crawlers import CURIACrawler
from data.database import CURIACaseDatabase

# Initialization
crawler = CURIACrawler() 
db = CURIACaseDatabase('data/curia.db')
db.create_tables(remove_old=False)

cases = crawler.crawl_ecj_cases()
db.write_cases(cases)

def write_docs(case):
    docs = crawler.crawl_case_docs(case)
    db.write_docs(case, docs)

map(write_docs, cases)

