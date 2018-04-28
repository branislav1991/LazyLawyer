"""Performs document rendering. Currently supports
rendering pdfs (using GhostScript). Planning to also
include rendering of HTML files and others.
"""

import os
import subprocess

def _run_pdfrenderer(args, cwd):
    """Calls the renderer with the appropriate
    arguments.
    Input params:
    args: arguments for the renderer as a list.
    """
    if os.name == 'posix':
        app = 'gs'
    elif os.name == 'nt':
        app = 'gswin64c'

    subprocess.run([app] + args,
        env={'PATH': os.getenv('PATH'), 'TEMP': os.getenv('TEMP')}, cwd=cwd)


def render_pdf(filepath, outputfilename, resolution):
    """Render pdf file to a specific format.
    Input params:
    filepath: PDF document to render.
    outputfilename: Name of the output file. The output
    will be saved in the same directory as the input file.
    resolution: Resolution in DPI.
    """
    outputfilestr = '-o ' + outputfilename
    resolutionstr = '-r' + str(resolution)
    _, format = os.path.splitext(outputfilestr)[1:]
    if format == 'png':
        sdevicestr = '-sDEVICE=png16m'
        compression = ''
    elif format == 'tiff':
        sdevicestr = '-sDEVICE=tiff24nc'
        compression = '-sCompression=lzw'

    cwd, filename = os.path.split(filepath)
    args = ['-q',resolutionstr,sdevicestr,compression,outputfilestr,filename,'-c','quit']
    _run_pdfrenderer(args, cwd=cwd)