import os

def strip_js_window_open(js):
    """Strips the javascript window.open function from 
    a link.
    """
    function_start = js.find('window.open(')
    function_end = js.find(');')
    arguments = js[function_start:function_end]
    broken = arguments.split(',')
    link = broken[0].split('(')[1:]
    link = '('.join(link)
    link = link[1:-1]
    return link

def to_full_year(two_digit_year, threshold=80):
    """Converts a year given by two digits to a full
    year number.
    Input params:
    threshold: up to which ending the year should be thought
    of as 1900s.
    """
    ending = int(two_digit_year)
    return (1900+ending) if ending > threshold else (2000+ending) 

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