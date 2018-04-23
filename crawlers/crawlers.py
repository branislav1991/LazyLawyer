from bs4 import BeautifulSoup
import helpers
import json
import re
import requests

class Crawler:
    SETUP_FILE_PATH = 'crawlers/crawler_setup.json'

    def __init__(self):
        """Intialize the crawler with the setup file.
        """
        with open(Crawler.SETUP_FILE_PATH, 'r') as setup_file:
            setup_json = json.load(setup_file) 
            self.eu_case_law_url = setup_json['eu_case_law_url']

    def crawl(self, url):
        """Crawl a specific url and return the soup.
        """
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        return soup

class CURIACrawler(Crawler):
    def __init__(self):
        super().__init__()

    def ecj_cases_to_json(self, json_path):
        """Crawl ECJ cases and save descriptions and links to a json file.
        """
        html = self.crawl(self.eu_case_law_url)

        case_rows = html.body.find_all('tr')
        def parse_case(row):
            try:
                link = row.find('b').a
                url = helpers.strip_js_window_open(link['href'])
                name = link.text.strip()
                desc = row.find('i').text.strip()
                return {'url': url, 'name': name, 'desc': desc}
            except (AttributeError, TypeError):
                return None
        self.cases_dict = [parse_case(r) for r in case_rows if parse_case(r) is not None]
            
        with open(json_path, 'w', encoding='UTF-8') as jsonfile:
            json.dump(cases_dict, jsonfile)

    def load_ecj_cases_json(self, json_path):
        """If a json file with urls of ecj cases already exists,
        we load it for further processing.
        """
        with open(json_path, 'r', encoding='UTF-8') as jsonfile:
            self.cases_dict = json.load(jsonfile)

    def _get_doc(self, case):
        match = re.search('.*/(\d+)', case['name'])
        year = helpers.to_full_year(match.group(1))
        if year > 1997:
            html = self.crawl(case['url'])
            doc_url = html.find('a', {'id': 'mainForm:j_id56'})['href']
            html_doc = self.crawl(doc_url)
            try:
                all_docs_html = html_doc.find('table', {'class': 'detail_table_documents'}) \
                    .find('tbody').find_all('tr', {'class': 'table_document_ligne'})
            except AttributeError:
                return None

            def get_doc_desc(html_tr):
                name = html_tr.find('td', {'class': 'table_cell_doc'}).text.split('\n')[0]
                ecli = html_tr.find('span', {'class': 'outputEcli'}).text
                date = html_tr.find('td', {'class': 'table_cell_date'}).text
                parties = html_tr.find('td', {'class': 'table_cell_nom_usuel'}).text
                subject = html_tr.find('td', {'class': 'table_cell_links_curia'}).find('span', {'class': 'tooltipLink'}).text
                links_curia = html_tr.find('td', {'class': 'table_cell_links_eurlex'}) \
                    .find_all('img', {'title': 'View pdf documents'})
                links_curia = [x.parent['href'] for x in links_curia]
                links_eurlex = html_tr.find_all('td', {'class': 'table_cell_aff'})[1] \
                    .find_all('img', {'title': 'View pdf documents'})
                links_eurlex = [x.parent['href'] for x in links_eurlex]
                return {'name': name, 'ecli': ecli, 'date': date, 
                    'parties': parties, 'subject': subject, 'links_curia': links_curia,
                    'links_eurlex': links_eurlex}

            all_docs = [get_doc_desc(x) for x in all_docs_html]
            return all_docs
        else:
            return None

    def download_cases_docs(self, n_cases=0, skip_cases=0, pdf=True, html=False):
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

        selected_cases_dict = self.cases_dict[skip_cases:skip_cases+n_cases]
        files_dict = [self._get_doc(x) for x in selected_cases_dict if self._get_doc(x) is not None]
        print(len(files_dict))

if __name__ == '__main__':
    path_cases_json = 'data/cases_list.json'
    crawler = CURIACrawler() 
    #crawler.ecj_cases_to_csv(path_cases_json)
    crawler.load_ecj_cases_json(path_cases_json)
    crawler.download_cases_docs(n_cases=10, skip_cases=4000)