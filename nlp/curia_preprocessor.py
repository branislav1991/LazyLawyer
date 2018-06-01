import helpers
from nlp.helpers import combine_split_result
from itertools import chain
import re
import warnings

def remove_header(doc):
    content = re.split(r"(?i)gives the following[\S\s]*?judgment", doc)
    if len(content) < 2:
        warnings.warn('Not the correct format for judgments')
        content = content[0]
    else:
        content = ''.join(content[1:])

    return content

def normalize(doc):
    content = re.sub(r'[^A-Za-z0-9\s.\/-]', r'', doc)
    content = re.sub(r'\n *([0-9]+)[. ]*\n', r'', content) # remove paragraph numbers
    content = re.sub(r'\n', r' ', content)
    return content

def split_to_sentences(doc):
    sentences = re.split(r"(?<![A-Z])\.", doc)
    return sentences

def tokenize(sentence):
    tokens = re.findall(r"[\w'\/-]+", sentence) # find all words

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