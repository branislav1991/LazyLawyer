"""This script crawls the CURIA database and saves all cases
and relevant pdf document links for each case to the database.
"""

from crawlers.crawlers import CURIACrawler
from database.database import CURIACaseDatabase

# Initialization
crawler = CURIACrawler() 
try:
    db = CURIACaseDatabase()
    db.create_tables(remove_old=True)

    cases = crawler.crawl_ecj_cases()
    db.write_cases(cases)

    for case in cases:
        docs = crawler.crawl_case_docs(case)
        if docs is not None:
            db.write_docs(case, docs)
finally:
    db.close()

# from tqdm import tqdm
# import requests

# url = "http://download.thinkbroadband.com/10MB.zip"
# response = requests.get(url, stream=True)

# with open("10MB", "wb") as handle:
#     for data in tqdm(response.iter_content()):
#         handle.write(data)