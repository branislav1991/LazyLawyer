from bs4 import BeautifulSoup
import requests

def crawl(url):
    """Crawl a specific url and return the soup.
    """
    req = requests.get(url)
    req.encoding = 'utf-8' # force utf-8 encoding
    soup = BeautifulSoup(req.text, 'html.parser')
    return soup

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == requests.codes.ok:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        response.raise_for_status()

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