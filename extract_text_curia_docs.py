"""This script tries to extract text and high-level structure from
documents. It does this by batch-processing all documents belonging
to all cases. For HTML documents, a text parser is called.
PDF documents are converted to an image format first (e.g. tiff) and
then processed with tesseract-ocr.
"""

from database.database import CURIACaseDatabase
from documents import doc_textextractor, doc_renderer
import helpers
from pathlib import Path
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

def text_from_docs_for_case(case, docs):
    """Extract text from documents in a case.
    """
    output_format = 'txt' # txt is the only supported text document format
    folder_path = Path('documents/' + helpers.case_name_to_folder(case['name']))

    for doc in docs:
        doc_filename = str(doc['id']) + '.' + doc['format']
        doc_path = str(folder_path / doc_filename)
        output_filename = str(doc['id']) + '.' + output_format

        if doc['format'] == 'pdf':
            # first convert pdf to tiff image
            img_filename = str(doc['id']) + '.tiff'
            doc_renderer.render_doc(doc_path, img_filename, 300)

            # extract text from tiff
            doc_textextractor.extract_text(str(folder_path / img_filename), output_filename)
        elif doc['format'] == 'html':
            output_filename = str(doc['id']) + '.' + output_format
            doc_textextractor.extract_text(str(folder_path / doc_filename), output_filename)

for case in tqdm(cases):
    docs = db.get_docs_for_case(case, only_valid=True)
    if len(docs) > 0:
        text_from_docs_for_case(case, docs)