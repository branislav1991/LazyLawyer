from bs4 import BeautifulSoup
import os
import subprocess

def _run_ocr(filename, outputfilename, cwd):
    app = 'tesseract'

    psmstr = '--psm 3'
    base, _ = os.path.splitext(outputfilename)

    args = [filename, base, psmstr]
    completed_process = subprocess.run([app] + args, env={'PATH': os.getenv('PATH')}, 
        cwd=cwd, check=True)

def _from_html(filename, outputfilename, cwd):
    with open(os.path.join(cwd, filename), 'r') as file:
        html = BeautifulSoup(file, 'html.parser')
        txt = html.body.get_text(separator='\n')
        with open(os.path.join(cwd, outputfilename), 'w') as outputfile:
            outputfile.write(txt.strip())

def extract_text(filepath, outputfilename):
    """Extract text from document and generate txt.
    """
    # check output format
    outputfilestr = '-o ' + outputfilename
    _, format = os.path.splitext(outputfilestr)
    assert format == '.txt' # we only support txt

    cwd, filename = os.path.split(filepath)

    _, format = os.path.splitext(filepath)
    if format == '.html':
        _from_html(filename, outputfilename, cwd)
    elif format == '.tiff':
        _run_ocr(filename, outputfilename, cwd)