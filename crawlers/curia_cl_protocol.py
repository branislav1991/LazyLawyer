import helpers

def crawl_cases(html):
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
    cases_dict = [parse_case(r) for r in case_rows if parse_case(r) is not None]
    return cases_dict