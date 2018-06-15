import argparse
from docai.scripts.crawl_curia_cases_docs import crawl_curia_cases_docs
from docai.scripts.download_curia_docs import download_curia_docs
from docai.scripts.extract_text_curia_docs import extract_text_curia_docs
from docai.scripts.extract_keywords_curia import extract_keywords_curia

def run_crawling_pipeline(num_cases=-1):
    crawl_curia_cases_docs(num_cases=num_cases)
    download_curia_docs()
    extract_text_curia_docs()
    extract_keywords_curia()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run whole crawling pipeline up to document content saving')
    parser.add_argument('--num_cases', type=int, default=-1, help='only crawl a limited number of cases')

    args = parser.parse_args()
    run_crawling_pipeline(args.num_cases)
