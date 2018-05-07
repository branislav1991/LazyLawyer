from documents import doc_downloader
from database.database import CURIACaseDatabase
import json
from requests.exceptions import HTTPError
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

for case in tqdm(cases):
    docs = db.get_docs_for_case(case, only_valid=True)
    if len(docs) > 0:
        try:
            doc_downloader.download_docs_for_case(case, docs)
            db.write_download_error(case, 0)
        except HTTPError:
            db.write_download_error(case, 1)