from database.database import CURIACaseDatabase
from documents import doc_textextractor
import helpers
from pathlib import Path
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

def text_from_docs_for_case(case, docs):
    """Extract text from documents in a case.
    """
    output_format = 'json' # json is the only supported text document format
    folder_path = Path('documents/' + helpers.case_name_to_folder(case['name']))
    for doc in docs:
        doc_filename = str(doc['id']) + '.' + doc['format']
        output_filename = str(doc['id']) + '.' + output_format

        doc_textextractor.extract_text(str(folder_path / doc_filename), output_filename)

for case in tqdm(cases):
    docs = db.get_docs_for_case(case, only_valid=True)
    if len(docs) > 0:
        text_from_docs_for_case(case, docs)