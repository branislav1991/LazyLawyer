import numpy as np

def combine_split_result(s):
    """Combines the result of a re.split splitting
    with the splitters retained. E.g. ['I', '/', 'am']
    -> ['I', '/am'].
    """
    if len(s) > 1:
        ss = [s[0]]
        for i in range(1, len(s)//2+1):
            ii = i-1
            ss.append(s[i] + s[i+1])
        return ss
    else:
        return s
