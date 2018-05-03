from database.database import CURIACaseDatabase
from documents import doc_renderer
import helpers
from pathlib import Path
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

def render_doc(document_path, output_filename, format, resolution):
    if format == 'pdf':
        pass
        #doc_renderer.render_pdf(document_path, output_filename, resolution)
    elif format == 'html':
        doc_renderer.render_html(document_path, output_filename)
    else:
        raise ValueError('Unsupported document format')

def render_docs_for_case(case, docs, output_format, resolution):
    """Render documents in a case to images.
    Input params:
    output_format: 'png' or 'tiff'.
    resolution: Output resolution in DPI.
    """
    folder_path = Path('documents/' + helpers.case_name_to_folder(case['name']))
    for doc in docs:
        doc_filename = str(doc['id']) + '.' + doc['format']
        output_filename = str(doc['id']) + '.' + output_format
        render_doc(str(folder_path / doc_filename), output_filename,
            doc['format'], resolution)

for case in tqdm(cases):
    docs = db.get_docs_for_case(case, only_valid=True)
    if len(docs) > 0:
        render_docs_for_case(case, docs, 'tiff', 300)