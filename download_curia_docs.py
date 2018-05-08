"""This script downloads documents from the curia database
and saves these documents in the doc_dir folder under the
case name and with a unique identifier as the document name.
"""
import concurrent.futures
from documents import doc_downloader
from database.database import CURIACaseDatabase
import helpers
import json
from requests.exceptions import HTTPError
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

def get_and_download_docs(case):
    # in this step we also skip documents which have already been downloaded
    docs = db.get_docs_for_case(case, only_with_link=True, downloaded=False)
    if len(docs) > 0:
        doc_downloader.download_docs_for_case(case, docs)

def main():
    for case in tqdm(cases):
        get_and_download_docs(case)
        db.write_download_error(case, 0)

if __name__ == '__main__':
    main()

# for case in tqdm(cases):
#     docs = db.get_docs_for_case(case, only_valid=True)
#     if len(docs) > 0:
#         try:
#             doc_downloader.download_docs_for_case(case, docs)
#             db.write_download_error(case, 0)
#         except HTTPError:
#             db.write_download_error(case, 1)