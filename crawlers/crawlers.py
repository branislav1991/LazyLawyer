from bs4 import BeautifulSoup
import helpers
import json
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
        self.html = self.crawl(self.eu_case_law_url)

    def ecj_cases_to_csv(self, json_path):
        """Crawl ECJ cases and save descriptions and links to a csv file.
        """
        cases_dict = []
        case_rows = self.html.body.find_all('tr')
        for row in case_rows:
            try:
                link = row.find('b').a
                url = helpers.strip_js_window_open(link['href'])
                name = link.text.strip()
                desc = row.find('i').text.strip()
                cases_dict.append({'url': url, 'name': name, 'desc': desc})
            except (AttributeError, TypeError):
                pass
        with open(json_path, 'w', encoding='UTF-8') as jsonfile:
            field_names = ['url', 'name', 'desc']
            json.dump(cases_dict, jsonfile)

if __name__ == '__main__':
    path_cases_json = 'data/cases_list.json'
    crawler = CURIACrawler() 
    crawler.ecj_cases_to_csv(path_cases_json)