from documents import doc_downloader
from database.database import CURIACaseDatabase
import json
from requests.exceptions import HTTPError
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

bad_cases = [] # docs that could not be downloaded properly
bad_cases_filepath = 'documents/bad_cases.txt'

for case in tqdm(cases):
    docs = db.get_docs_for_case(case)
    if len(docs) > 0:
        try:
            doc_downloader.download_docs_for_case(case, docs)
        except HTTPError:
            bad_cases.append(case)

# document bad cases
if len(bad_cases) > 0:
    with open(bad_cases_filepath, 'w') as file:
        json.dump(bad_cases, file)