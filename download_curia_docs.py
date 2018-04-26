from documents import doc_downloader
from database.database import CURIACaseDatabase

db = CURIACaseDatabase()
cases = db.get_all_cases()

for case in cases:
    docs = db.get_docs_for_case(case)
    if len(docs) > 0:
        doc_downloader.download_docs_for_case(case, docs)
    