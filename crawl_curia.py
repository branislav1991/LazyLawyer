"""This script crawls the CURIA database and saves all cases
and relevant pdf document links for each case to the database.
"""

from crawlers.crawlers import CURIACrawler
from database.database import CURIACaseDatabase
from tqdm import tqdm

crawl_docs_only = False
crawl_pdf = True
crawl_html = False

crawler = CURIACrawler() 

try:
    db = CURIACaseDatabase()
    if crawl_docs_only:
        cases = db.get_all_cases() 
        max_case_id = db.get_max_case_id_in_docs()
        cases = [x for x in cases if x['id'] > max_case_id]

    else:
        cases = crawler.crawl_ecj_cases()
        db.write_cases(cases)

    for case in tqdm(cases):
        docs = crawler.crawl_case_docs(case, crawl_pdf, crawl_html)
        if docs is not None:
            db.write_docs(case, docs)

finally:
    db.close()