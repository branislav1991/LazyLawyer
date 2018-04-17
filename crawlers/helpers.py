def strip_js_window_open(js):
    """Strips the javascript window.open function from 
    a link.
    """
    function_start = js.find('window.open(')
    function_end = js.find(');')
    arguments = js[function_start:function_end]
    broken = arguments.split(',')
    link = broken[0].split('(')[1]
    link = link[1:-1]
    return link