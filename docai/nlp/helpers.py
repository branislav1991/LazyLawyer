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

def split_to_ngrams(word, n_min=3, n_max=6):
    """Splits word into various ngrams it contains.
    You can specify the minimum and maximum ngram
    size by n_min and n_max params. This function
    also returns the original word as part of the list.
    """
    word_ngrams = [word]
    if len(word) > n_min:
        for n in range(n_min, n_max+1):
            ngrams = [word[i:i+n] for i in range(0, len(word)-n+1)]
            word_ngrams.extend(ngrams)
    return word_ngrams
