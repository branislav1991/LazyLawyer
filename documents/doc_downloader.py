import helpers
import requests
import os
from tqdm import tqdm

def _download_file(url, filename):
    testread = requests.head(url_r) # A HEAD request only downloads the headers
    filelength = int(testread.headers['Content-length'])

    r = requests.get(url, stream=True) # actual download full file

    with open(filename, 'wb') as f:
        pbar = tqdm(total=int(filelength/1024))
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                pbar.update ()
                f.write(chunk)

def download_document(name, case, type, url):
    """Downloads documents from the web belonging to a
    specific case and type (e.g. pdf). Stores the document
    under [name].[type].
    """
    folder_path = os.path.join('documents', case, type)
    helpers.create_folder_if_not_exists(folder_path)  
    _download_file(url, os.path.join(folder_path, '.'.join(name, type)))
