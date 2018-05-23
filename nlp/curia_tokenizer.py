import helpers
from nlp.helpers import combine_split_result
from itertools import chain
import re
import warnings

def tokenize(doc):
    # remove header
    content = re.split(r"(?i)gives the following[\S\s]*?judgment", doc)
    if len(content) < 2:
        warnings.warn('Not the correct format for judgments')
        content = content[0]
    else:
        content = ''.join(content[1:])

    content = re.sub(r'[^A-Za-z0-9\s.\/-]', r'', content)

    content = re.sub(r'\n *([0-9]+)[. ]*\n', r'', content) # remove paragraph numbers

    content = re.sub(r'\n', r' ', content)

    tokens = re.findall(r"[\w'\/-]+", content) # find all words

    # remove numbers (preliminary only; of course Article numbers are important)
    #tokens = [[word for word in sent if not word[0].isdigit()] for sent in tokens] 

    # simple word splitting: split words that begin with capitals
    tokens_split = []
    for word in tokens:
        s = re.split(r"((?<=[a-z])[A-Z])", word)
        s = combine_split_result(s)
        tokens_split.extend(s)
    tokens = tokens_split

    tokens = [word.lower() for word in tokens] # lowercase words
    #tokens = [word for word in tokens if word not in self.stopwords] # remove stopwords
    return tokens