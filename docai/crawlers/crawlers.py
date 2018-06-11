from bs4 import BeautifulSoup
import docai.helpers
import docai.crawlers.helpers
import json
import re

class CURIACrawler:
    """Crawler class for crawling over all CURIA docs (eurlex DB).
    """
    def __init__(self):
        self.eu_case_law_links = docai.helpers.setup_json['eu_case_law_links']

    def crawl_ecj_cases(self, num_cases):
        """Crawl ECJ cases and save descriptions and links to a json file.
        Input params:
        num_cases: how many cases to crawl; if <= 0, all cases are crawled.
        """
        cases_dict = []
        for link in self.eu_case_law_links:
            html = docai.crawlers.helpers.crawl(link['url'])
            protocol = docai.helpers.import_by_name(link['protocol'])
            cases = protocol.crawl_cases(html)
            if num_cases > 0:
                cases = cases[:num_cases]

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
        case: case for which to crawl documents.
        formats: formats of docs to download (pdf, html).
        """
        match = re.search('.*/(\d+)', case['name'])
        year = docai.crawlers.helpers.to_full_year(match.group(1))

        if year > 1997:
            protocol = docai.helpers.import_by_name(case['protocol'])
            html = docai.crawlers.helpers.crawl(case['url'])

            docs = protocol.crawl_docs(html, formats)
            return docs
        else:
            return None
