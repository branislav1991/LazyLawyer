"""This script tries to extract text and high-level structure from
documents. It does this by batch-processing all documents belonging
to all cases. For HTML documents, a text parser is called.
PDF documents are converted to an image format first (e.g. tiff) and
then processed with tesseract-ocr.
"""

from docai.database import table_cases, table_docs, table_doc_contents
from docai.documents import doc_textextractor, doc_renderer
from docai import helpers
import os
from pathlib import Path
from tqdm import tqdm

def text_from_doc(doc):
    """Extract text from documents in a case.
    """
    case = table_docs.get_doc_case(doc)    
    folder_path = Path('doc_dir/' + helpers.case_name_to_folder(case['name']))

    doc_filename = str(doc['id']) + '.' + doc['format']
    doc_path = str(folder_path / doc_filename)

    text = None
    if doc['format'] == 'pdf':
        try:
            # first convert pdf to tiff image
            img_filename = 'tmp.tiff'
            doc_renderer.render_doc(doc_path, img_filename, 300)

            os.rename(str(folder_path / (' ' + img_filename)), str(folder_path / img_filename))

            # extract text from tiff and delete it
            text = doc_textextractor.extract_from_image(str(folder_path / img_filename))
        finally:
            if (os.path.exists(str(folder_path / img_filename))):
                os.remove(str(folder_path / img_filename))

    elif doc['format'] == 'html':
        text = doc_textextractor.extract_from_html(str(folder_path / doc_filename))

    return text

def main():
    cases = table_cases.get_all_cases()

    docs = table_docs.get_docs_with_name('Judgment', only_valid=True)
    if len(docs) > 0:
        for doc in docs:
            # first check if document could be downloaded
            if doc['download_error'] is None or doc['download_error'] > 0:
                continue

            if doc['content_id'] is None:
                text = text_from_doc(doc)
                if text is not None:
                    table_doc_contents.write_doc_content(doc, text)

if __name__ == '__main__':
    main()
