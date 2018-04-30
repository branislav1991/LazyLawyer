from bs4 import BeautifulSoup
import crawlers.curia_cl_protocol
import helpers
import crawlers.helpers
import json
import re

class Crawler:
    SETUP_FILE_PATH = 'crawlers/crawler_setup.json'

    def __init__(self):
        pass

class CURIACrawler(Crawler):
    def __init__(self):
        with open(Crawler.SETUP_FILE_PATH, 'r') as setup_file:
            setup_json = json.load(setup_file) 
            self.eu_case_law_links = setup_json['eu_case_law_links']

    def crawl_ecj_cases(self):
        """Crawl ECJ cases and save descriptions and links to a json file.
        """
        cases_dict = []
        for link in self.eu_case_law_links:
            html = self._crawl(link['url'])
            protocol = helpers.import_by_name(link['protocol'])
            cases = protocol.crawl_cases(html)
            for c in cases: # append protocol to each case to know how it was crawled
                c.update({'protocol': link['protocol']})

            cases_dict.extend(cases)
        
        return cases_dict

    def crawl_case_docs(self, case, formats):
        """Crawl individual cases from the case directory.
        Requires the case dictionary to be already loaded either
        by calling ecj_cases_to_json() or load_ecj_cases_json().
        Currently, cases from 1997 and older are ignored.
        Input params:
        n_cases: how many cases to scrape. If 0, all cases are scraped (default=0).
        skip_cases: how many cases to skip at the beginning (default=0).
        formats: formats of docs to download (pdf, html).
        """
        match = re.search('.*/(\d+)', case['name'])
        year = crawlers.helpers.to_full_year(match.group(1))

        if year > 1997:
            protocol = helpers.import_by_name(case['protocol'])
            html = crawlers.helpers.crawl(case['url'])

            docs = protocol.crawl_docs(html, formats)
            return docs
        else:
            return None
