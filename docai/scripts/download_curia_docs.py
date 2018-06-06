"""This script downloads documents from the curia database
and saves these documents in the doc_dir folder under the
case name and with a unique identifier as the document name.
"""
import concurrent.futures
from docai.documents import doc_downloader
from docai.database import table_cases, table_docs
from docai import helpers
import json
from requests.exceptions import HTTPError
from tqdm import tqdm

def get_and_download_docs(case):
    # in this step we also skip documents which have already been downloaded
    docs = table_docs.get_docs_for_case(case, only_with_link=True, downloaded=False)
    for doc in docs:
        try:
            doc_downloader.download_doc_for_case(case, doc)
            table_docs.write_download_error(doc, 0)
        except HTTPError:
            table_docs.write_download_error(doc, 1)

def main():
    cases = table_cases.get_all_cases()

    for case in tqdm(cases):
        get_and_download_docs(case)

if __name__ == '__main__':
    main()
