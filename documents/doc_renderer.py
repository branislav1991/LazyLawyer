"""Performs document rendering. Currently supports
rendering pdfs (using GhostScript). Planning to also
include rendering of HTML files and others.
"""

import os
import subprocess

def _run_pdfrenderer(args):
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
        env={'PATH': os.getenv('PATH'), 'TEMP': os.getenv('TEMP')})


def render_pdf(filepath, outputfilepath, format, resolution):
    """Render pdf file to a specific format.
    Input params:
    filepath: PDF document to render.
    outputfilepath: Where to save the output.
    format: 'png' or 'tiff'.
    resolution: Resolution in DPI.
    """
    outputfilestr = '-o ' + outputfilepath
    resolutionstr = '-r' + str(resolution)
    if format == 'png':
        sdevicestr = '-sDEVICE=png16m'
        compression = ''
    elif format == 'tiff':
        sdevicestr = '-sDEVICE=tiff24nc'
        compression = '-sCompression=lzw'

    args = ['-q',resolutionstr,sdevicestr,compression,outputfilestr,filepath,'-c','quit']
    _run_pdfrenderer(args)

if __name__ == '__main__':
    render_pdf('documents/hello.pdf', 'a.tiff', 'tiff', 300)