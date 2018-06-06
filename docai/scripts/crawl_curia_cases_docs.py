"""This script crawls the CURIA database and saves all cases
and relevant document links for each case to the database.
"""
import concurrent.futures
from docai.crawlers.crawlers import CURIACrawler
from docai.database import table_cases, table_docs
from docai import helpers
from tqdm import tqdm

def main():
    crawl_docs_only = False # if this is true, only docs are crawled instead of cases and docs
    formats = ['html', 'pdf'] # formats are processed in the order they are given

    crawler = CURIACrawler() 

    if crawl_docs_only:
        cases = table_cases.get_all_cases() 
        max_case_id = table_docs.get_max_case_id_in_docs()
        cases = [x for x in cases if x['id'] > max_case_id]
    else:
        cases = crawler.crawl_ecj_cases()
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
    main()
