from documents import doc_downloader
from database.database import CURIACaseDatabase
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

for case in tqdm(cases):
    docs = db.get_docs_for_case(case)
    if len(docs) > 0:
        doc_downloader.download_docs_for_case(case, docs)
    