from crawlers.crawlers import CURIACrawler
from data import database

crawler = CURIACrawler() 

crawler.crawl_ecj_cases(path_cases_json)
crawler.download_cases_docs(n_cases=10, skip_cases=4000)