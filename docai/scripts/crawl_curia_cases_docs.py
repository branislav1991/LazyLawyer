"""This script crawls the CURIA database and saves all cases
and relevant document links for each case to the database.
"""
import argparse
import concurrent.futures
from docai.crawlers.crawlers import CURIACrawler
from docai.database import table_cases, table_docs
from docai import helpers
from tqdm import tqdm

def crawl_curia_cases_docs(crawl_docs_only=False, num_cases=-1):
    """Crawls cases and the corresponding documents.
    Input params:
    crawl_docs_only: If True, does not crawl cases and only crawls docs.
    num_cases: If <= 0, crawls all available cases. Otherwise, crawls num_cases
    cases.
    """
    formats = ['html', 'pdf'] # formats are processed in the order they are given

    crawler = CURIACrawler() 

    if crawl_docs_only:
        cases = table_cases.get_all_cases() 
        max_case_id = table_docs.get_max_case_id_in_docs()
        cases = [x for x in cases if x['id'] > max_case_id]
    else:
        cases = crawler.crawl_ecj_cases(num_cases)
        table_cases.write_cases(cases)

    cases_batches = helpers.create_batches_list(cases, 50)
    for batch in tqdm(cases_batches):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures_cases = {executor.submit(crawler.crawl_case_docs, case, formats):case for case in batch}

        for future in concurrent.futures.as_completed(futures_cases):
            case = futures_cases[future]
            docs = future.result()
            if docs is not None:
                table_docs.write_docs_for_case(case, docs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl CURIA cases and docs.')
    parser.add_argument('--docs_only', action='store_true', help='only crawl documents')
    parser.add_argument('--num_cases', type=int, default=-1, help='only crawl a limited number of cases')

    args = parser.parse_args()
    crawl_cases_docs(args.docs_only, args.num_cases)
