import lazylawyer.crawlers.crawlers
import pytest
import lazylawyer.helpers
import lazylawyer.crawlers.helpers
import lazylawyer.scripts.run_crawl_pipeline
import lazylawyer.scripts.migrate_db
import lazylawyer.database.table_doc_contents
import lazylawyer.database.table_docs
import os

def test_import():
    for link in lazylawyer.helpers.setup_json['eu_case_law_links']:
        lazylawyer.helpers.import_by_name(link['protocol'])

def test_create_batches_list():
    lst = list(range(0,10))
    batch_size = 3
    batches = lazylawyer.helpers.create_batches_list(lst, batch_size)
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]

def test_create_batches_generate():
    lst = list(range(0,10))
    batch_size = 3
    batches = lazylawyer.helpers.create_batches_generate(lst, batch_size)
    assert next(batches) == [0, 1, 2]
    assert next(batches) == [3, 4, 5]
    assert next(batches) == [6, 7, 8]
    assert next(batches) == [9]

def test_case_name_to_folder():
    case_name = 'Hello/World/'
    folder_name = lazylawyer.helpers.case_name_to_folder(case_name)
    assert folder_name == 'Hello_World_'

def test_case_folder_to_name():
    folder_name = 'Hello_World_'
    case_name = lazylawyer.helpers.case_folder_to_name(folder_name)
    assert case_name == 'Hello/World/'

def test_to_full_year():
    year = '81'
    full_year = lazylawyer.crawlers.helpers.to_full_year(year)
    assert full_year == 1981
    year = '32'
    full_year = lazylawyer.crawlers.helpers.to_full_year(year)
    assert full_year == 2032

def test_crawl_ecj_cases():
    # this test just tests if a large number of cases was
    # downloaded and if no error occured
    crawler = lazylawyer.crawlers.crawlers.CURIACrawler()
    cases_dict, appeals_dict = crawler.crawl_ecj_cases()
    assert len(cases_dict) > 28000

def test_crawling_pipeline():
    # runs the whole crawling pipeline from obtaining cases through
    # downloading documents to extracting text and saving it in the
    # database. It asserts, if content for the documents named 'Judgment'
    # is available in the 'doc_contents' table.
    lazylawyer.scripts.migrate_db.migrate_db()
    lazylawyer.scripts.run_crawl_pipeline.run_crawl_pipeline(num_cases=1250)
    docs = lazylawyer.database.table_docs.get_docs_with_names(['Judgment'])
    doc_contents = [lazylawyer.database.table_doc_contents.get_doc_content(doc) for doc in docs]
    assert len(doc_contents) == 19
    for content in doc_contents:
        assert len(content) > 0
