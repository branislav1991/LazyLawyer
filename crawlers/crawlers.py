from bs4 import BeautifulSoup
import crawlers.curia_cl_protocol
import helpers
import json
import re
import requests

class Crawler:
    SETUP_FILE_PATH = 'crawlers/crawler_setup.json'

    def __init__(self):
        pass

    def _crawl(self, url):
        """Crawl a specific url and return the soup.
        """
        req = requests.get(url)
        req.encoding = 'utf-8' # force utf-8 encoding
        soup = BeautifulSoup(req.text, 'html.parser')
        return soup

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
            result = protocol.crawl_cases(html)
            cases_dict.extend(result)
        
        return cases_dict

    def crawl_case_docs(self, case, pdf, html):
        """Crawl individual cases from the case directory.
        Requires the case dictionary to be already loaded either
        by calling ecj_cases_to_json() or load_ecj_cases_json().
        Currently, cases from 1997 and older are ignored.
        Input params:
        n_cases: how many cases to scrape. If 0, all cases are scraped (default=0).
        skip_cases: how many cases to skip at the beginning (default=0).
        pdf: download all case-relevant pdfs (defalut=True).
        html: download all case-relevant html documents (default=False).
        """
        match = re.search('.*/(\d+)', case['name'])
        year = helpers.to_full_year(match.group(1))
        if year > 1997:
            html = self._crawl(case['url'])
            doc_url = html.find('a', {'id': 'mainForm:j_id56'})
            doc_url = doc_url['href']
            html_doc = self._crawl(doc_url)
            try:
                all_docs_html = html_doc.find('table', {'class': 'detail_table_documents'}) \
                    .find('tbody').find_all('tr', {'class': 'table_document_ligne'})
            except AttributeError:
                return None

            def get_doc_desc(html_tr):
                name = html_tr.find('td', {'class': 'table_cell_doc'})
                name = None if name is None else name.text.split('\n')[0]
                ecli = html_tr.find('span', {'class': 'outputEcli'})
                ecli = None if ecli is None else ecli.text
                date = html_tr.find('td', {'class': 'table_cell_date'})
                date = None if date is None else date.text
                parties = html_tr.find('td', {'class': 'table_cell_nom_usuel'})
                parties = None if parties is None else parties.text
                subject = html_tr.find('td', {'class': 'table_cell_links_curia'}).find('span', {'class': 'tooltipLink'})
                subject = None if subject is None else subject.text

                def link_to_image(imgs_list):
                    try:
                        links = [x.parent['href'] for x in imgs_list]
                        link =  None if len(links) < 1 else links[0]
                    except KeyError:
                        link = None
                    return link

                if pdf:
                    links_curia = html_tr.find('td', {'class': 'table_cell_links_eurlex'}) \
                        .find_all('img', {'title': 'View pdf documents'})
                    link_curia = link_to_image(links_curia)

                    links_eurlex = html_tr.find_all('td', {'class': 'table_cell_aff'})[1] \
                        .find_all('img', {'title': 'View pdf documents'})
                    link_eurlex = link_to_image(links_eurlex)
                
                type = 'pdf'

                return {'name': name, 'ecli': ecli, 'date': date, 
                    'parties': parties, 'subject': subject, 'link_curia': link_curia,
                    'link_eurlex': link_eurlex, 'type': type}

            all_docs = [get_doc_desc(x) for x in all_docs_html]
            return all_docs
        else:
            return None
