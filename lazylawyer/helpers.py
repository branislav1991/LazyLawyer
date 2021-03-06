import json
import os
import requests

SETUP_FILE_PATH = os.path.join('lazylawyer', 'setup.json')
with open(SETUP_FILE_PATH, 'r') as setup_file:
    setup_json = json.load(setup_file) 

def import_by_name(name):
    """Imports a python module by its
    qualified name.
    """
    module = __import__(name)
    for n in name.split(".")[1:]:
        module = getattr(module, n)
    return module

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == requests.codes.ok:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        response.raise_for_status()

def create_batches_generate(lst, batch_size):
    """Generate batches using a generator.
    """
    # For item i in a range that is a length of l,
    for i in range(0, len(lst), batch_size):
        # Create an index range for l of n items:
        yield lst[i:i+batch_size]

def create_batches_list(lst, batch_size):
    """Return list of batches.
    """
    batches = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    return batches

def create_folder_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def case_name_to_folder(name):
    """Convert name of case to case folder."""
    return name.replace('/', '_')

def case_folder_to_name(folder_name):
    """Convert name of case folder to case name."""
    return folder_name.replace('_', '/')
