"""Performs document rendering. Currently supports
rendering pdfs (using GhostScript). Planning to also
include rendering of HTML files and others.
"""

import asyncio
import os
from pyppeteer import launch
import subprocess

def render_html(file_path, output_filename):
    """Render html file to a specific format.
    Input params:
    file_path: HTML document to render.
    output_path: Name of the output file. The output
    will be saved in the same directory as the input file.
    """
    output_noext, format = os.path.splitext(output_filename)
    if (format != '.png'):
        print('Forcing output format of html file to png')
        format = '.png'

    path, filename = os.path.split(file_path)
    output_path = os.path.join(path, output_noext + format)

    async def make_screenshot():
        browser = await launch()
        page = await browser.newPage()
        await page.goto(file_path)
        await page.screenshot({'path': output_path, 'fullPage': True})
        await browser.close()

    asyncio.get_event_loop().run_until_complete(make_screenshot())

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

    completed_process = subprocess.run([app] + args, env={'PATH': os.getenv('PATH'), 'TEMP': os.getenv('TEMP')}, 
        cwd=cwd, check=True)

def render_pdf(file_path, output_filename, resolution):
    """Render pdf file to a specific format.
    Input params:
    file_path: PDF document to render.
    output_name: Name of the output file. The output
    will be saved in the same directory as the input file.
    resolution: Resolution in DPI.
    """
    outputfilestr = '-o ' + output_filename
    resolutionstr = '-r' + str(resolution)
    _, format = os.path.splitext(output_filename)
    if format == '.png':
        sdevicestr = '-sDEVICE=png16m'
        compression = ''
    elif format == '.tiff':
        sdevicestr = '-sDEVICE=tiff24nc'
        compression = '-sCompression=lzw'
    else:
        raise ValueError('Unsupported format')

    cwd, filename = os.path.split(file_path)
    args = ['-q',resolutionstr,sdevicestr,compression,outputfilestr,filename,'-c','quit']
    _run_pdfrenderer(args, cwd=cwd)

def render_doc(document_path, output_filename, format, resolution):
    if format == 'pdf':
        render_pdf(document_path, output_filename, resolution)
    elif format == 'html':
        render_html(document_path, output_filename)
    else:
        raise ValueError('Unsupported document format')