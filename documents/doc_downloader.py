import helpers
import os
from pathlib import Path
import requests
from tqdm import tqdm

def _download_file(url, filename):
    testread = requests.head(url) # A HEAD request only downloads the headers
    filelength = int(testread.headers['Content-length'])

    r = requests.get(url, stream=True) # actual download full file

    with open(filename, 'wb') as f:
        pbar = tqdm(total=int(filelength/1024))
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                pbar.update ()
                f.write(chunk)

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