from bs4 import BeautifulSoup
import os
import subprocess
import time

def extract_from_image(file_path):
    app = 'tesseract'

    psmstr = '--psm 3'
    cwd, filename = os.path.split(file_path)
    base, _ = os.path.splitext(filename)
    outputfilename = base + '.txt'
    outputfile_path = os.path.join(cwd, outputfilename)

    args = [filename, base, psmstr]

    try:
        completed_process = subprocess.run([app] + args, env={'PATH': os.getenv('PATH')}, 
            cwd=cwd, check=True)
        with open(outputfile_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()
            text = text.strip()
    finally: 
        # delete temporary text file
        if (os.path.exists(outputfile_path)):
            os.remove(outputfile_path)
    return text

def extract_from_html(file_path):
    with open(os.path.join(file_path), 'r') as file:
        html = BeautifulSoup(file, 'html.parser')
        text = html.body.get_text(separator='\n')
        text = text.strip()
    return text