import helpers
import os
from pathlib import Path
import requests

def _download_file(url, filename):
    response = requests.get(url)
    if response.status_code == requests.codes.ok:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        response.raise_for_status()

def download_docs_for_case(case, docs):
    """Downloads documents from the web belonging to a
    specific case. Stores the document
    under [name].[type].
    """
    folder_path = Path('documents/' + helpers.case_name_to_folder(case['name']))
    helpers.create_folder_if_not_exists(folder_path)
    for doc in docs:
        doc_filename = str(doc['id']) + '.' + doc['type']
        if doc['link_curia'] is not None:
            _download_file(doc['link_curia'], folder_path / doc_filename)
        if doc['link_eurlex'] is not None:
            _download_file(doc['link_eurlex'], folder_path / doc_filename)