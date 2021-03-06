from contextlib import suppress
import lazylawyer.helpers
import lazylawyer.crawlers.helpers
import itertools
import re

def _link_to_image(imgs_list):
    try:
        links = [x.parent['href'] for x in imgs_list]
        link =  None if len(links) < 1 else links[0]
    except KeyError:
        link = None
    return link

def _crawl_doc(html_tr, formats):
    """Process one doc.
    """
    name = html_tr.find('td', {'class': 'table_cell_doc'})
    name = None if name is None else name.text.split('\n')[0]
    ecli = html_tr.find('span', {'class': 'outputEcli'})
    ecli = None if ecli is None else ecli.text
    date = html_tr.find('td', {'class': 'table_cell_date'})
    date = None if date is None else date.text
    party1 = html_tr.find('td', {'class': 'table_cell_nom_usuel'})
    party1 = None if party1 is None else party1.text.strip()
    party2 = None
    if party1 is not None:
        parties = re.match(r'(.*) v (.*)', party1)
        if parties and parties.group(1) and parties.group(2):
            party1 = parties.group(1)
            party2 = parties.group(2)

    subject = html_tr.find('td', {'class': 'table_cell_links_curia'}).find('span', {'class': 'tooltipLink'})
    subject = None if subject is None else subject.text

    source = None
    format = None
    sources = ['curia', 'eurlex']
    for fs in itertools.product(formats, sources):
        # Try to get documents in the formats given. First format
        # in the list gets precedence over the second etc.
        # Similarly, the hardcoded sources are scanned from first to last.
        link = None
        if (fs[0] == 'pdf'):
            if (fs[1] == 'curia'):
                links_curia = html_tr.find('td', {'class': 'table_cell_links_eurlex'}) \
                    .find_all('img', {'title': 'View pdf documents'})
                link = _link_to_image(links_curia)
            elif (fs[1] == 'eurlex'):
                links_eurlex = html_tr.find_all('td', {'class': 'table_cell_aff'})[1] \
                    .find_all('img', {'title': 'View pdf documents'})
                link = _link_to_image(links_eurlex)

        elif (fs[0] == 'html'):
            if (fs[1] == 'curia'):
                links_curia = html_tr.find('td', {'class': 'table_cell_links_eurlex'}) \
                    .find_all('img', {'title': 'View html documents'})
                doc_url = _link_to_image(links_curia)
                if doc_url is not None: # if a document exists
                    with suppress(Exception): # if we fail, we might as well search further
                        html_doc = lazylawyer.crawlers.helpers.crawl(doc_url)
                        link = html_doc.find('a', {'id': 'mainForm:j_id159'})['href']

            elif (fs[1] == 'eurlex'):
                links_eurlex = html_tr.find_all('td', {'class': 'table_cell_aff'})[1] \
                    .find_all('img', {'title': 'View html documents'})
                link = _link_to_image(links_eurlex)
                
        if link is not None: # stop iterating if we found a link to the document
            format = fs[0]
            source = fs[1]
            break

    return {'name': name, 'ecli': ecli, 'date': date, 
        'party1': party1, 'party2': party2, 'subject': subject, 'link': link,
        'source': source, 'format': format}

def crawl_cases(html):
    case_rows = html.body.find_all('tr')
    def parse_case(row):
        try:
            link = row.find('b').a
            url = lazylawyer.crawlers.helpers.strip_js_window_open(link['href'])
            name = link.text.strip()
            desc = row.find('i').text.strip()
            court = 'GC' if name.startswith('T') else 'COJ'
            return {'url': url, 'name': name, 'desc': desc, 'court': court}
        except (AttributeError, TypeError):
            return None

    def parse_appeal(row):
        try:
            link = row.find('b').a
            name = link.text.strip()
            appeal = row.find('i').find(text='APPEAL : ')
            appeal = appeal.parent.findNext('a').text
            return {'orig': name, 'appeal': appeal}
        except (AttributeError, TypeError):
            return None

    cases_dict = [parse_case(r) for r in case_rows if parse_case(r) is not None]
    appeals_dict = [parse_appeal(r) for r in case_rows if parse_appeal(r) is not None]
    return cases_dict, appeals_dict

def crawl_docs(html, formats):
    """Crawl docs for a specific case.
    """
    doc_url = html.find('a', {'id': 'mainForm:j_id56'})
    doc_url = doc_url['href']
    html_doc = lazylawyer.crawlers.helpers.crawl(doc_url)
    try:
        all_docs_html = html_doc.find('table', {'class': 'detail_table_documents'}) \
            .find('tbody').find_all('tr', {'class': 'table_document_ligne'})
    except AttributeError:
        return None

    all_docs = [_crawl_doc(x, formats) for x in all_docs_html]
    return all_docs
