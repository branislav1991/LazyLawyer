import helpers
import os
from pathlib import Path
import requests

def download_docs_for_case(case, docs):
    """Downloads documents from the web belonging to a
    specific case. Stores the document
    under [name].[format].
    """
    folder_path = Path('documents/' + helpers.case_name_to_folder(case['name']))
    if os.path.exists(folder_path):
        # we assume that the documents already exists; skip
        return

    os.makedirs(folder_path)
    for doc in docs:
        doc_filename = str(doc['id']) + '.' + doc['format']
        if doc['link'] is not None:
            helpers.download_file(doc['link'], folder_path / doc_filename)