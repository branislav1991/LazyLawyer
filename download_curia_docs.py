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
    docs = db.get_docs_for_case(case, only_valid=True)
    if len(docs) > 0:
        doc_downloader.download_docs_for_case(case, docs)

def main():
    cases_batches = helpers.create_batches_list(cases, 10)
    for batch in tqdm(cases_batches):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures_cases = {executor.submit(get_and_download_docs, case):case for case in batch}

        for future in concurrent.futures.as_completed(futures_cases):
            case = futures_cases[future]
            try:
                docs = future.result()
                db.write_download_error(case, 0)
            except HTTPError:
                db.write_download_error(case, 1)

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