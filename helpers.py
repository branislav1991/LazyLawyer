import os

def import_by_name(name):
    """Imports a python module by its
    qualified name.
    """
    module = __import__(name)
    for n in name.split(".")[1:]:
        module = getattr(module, n)
    return module

def create_batches(list, batch_size):
    # For item i in a range that is a length of l,
    for i in range(0, len(list), batch_size):
        # Create an index range for l of n items:
        yield list[i:i+batch_size]

def create_folder_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def case_name_to_folder(name):
    """Convert name of case to case folder."""
    return name.replace('/', '_')

def case_folder_to_name(folder_name):
    """Convert name of case folder to case name."""
    return folder_name.replace('_', '/')