from database.database import CURIACaseDatabase
from documents import doc_renderer
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

def render_docs_for_case(case, docs, format, resolution):
    folder_path = Path('documents/' + helpers.case_name_to_folder(case['name']))
    for doc in docs:
        doc_filename = str(doc['id']) + '.' + doc['type']
        output_filename = str(doc['id']) + '.' + format
        doc_renderer.render_pdf(folder_path / doc_filename, folder_path / output_filename,
            resolution)

for case in tqdm(cases):
    docs = db.get_docs_for_case(case)
    if len(docs) > 0:
        render_docs_for_case(case, docs)