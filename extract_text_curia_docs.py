"""This script tries to extract text and high-level structure from
documents. It does this by batch-processing all documents belonging
to all cases. For HTML documents, a text parser is called.
PDF documents are converted to an image format first (e.g. tiff) and
then processed with tesseract-ocr.
"""

from database.database import CURIACaseDatabase
from documents import doc_textextractor, doc_renderer
import helpers
import os
from pathlib import Path
from tqdm import tqdm

db = CURIACaseDatabase()
cases = db.get_all_cases()

def text_from_docs(docs):
    """Extract text from documents in a case.
    """
    output_format = 'txt' # txt is the only supported text document format
    for doc in docs:
        case = db.get_doc_case(doc)    
        folder_path = Path('doc_dir/' + helpers.case_name_to_folder(case['name']))

        doc_filename = str(doc['id']) + '.' + doc['format']
        doc_path = str(folder_path / doc_filename)
        output_filename = str(doc['id']) + '.' + output_format

        if doc['format'] == 'pdf':
            try:
                # first convert pdf to tiff image
                img_filename = 'tmp.tiff'
                doc_renderer.render_doc(doc_path, img_filename, 300)

                os.rename(str(folder_path / (' ' + img_filename)), str(folder_path / img_filename))

                # extract text from tiff and delete it
                doc_textextractor.extract_text(str(folder_path / img_filename), output_filename)
            finally:
                if (os.path.exists(str(folder_path / img_filename))):
                    os.remove(str(folder_path / img_filename))

        elif doc['format'] == 'html':
            output_filename = str(doc['id']) + '.' + output_format
            doc_textextractor.extract_text(str(folder_path / doc_filename), output_filename)

docs = db.get_docs_with_name('Judgment', only_valid=True)
if len(docs) > 0:
    text_from_docs(docs)