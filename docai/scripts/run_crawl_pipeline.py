import argparse
from docai.scripts.crawl_cases_docs_curia import crawl_cases_docs_curia
from docai.scripts.download_docs_curia import download_docs_curia
from docai.scripts.extract_content_curia import extract_content_curia

def run_crawl_pipeline(num_cases=-1):
    crawl_cases_docs_curia(num_cases=num_cases)
    download_docs_curia()
    extract_content_curia()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run whole crawling pipeline up to document content saving')
    parser.add_argument('--num_cases', type=int, default=-1, help='only crawl a limited number of cases')

    args = parser.parse_args()
    run_crawl_pipeline(args.num_cases)
