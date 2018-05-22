import helpers
from nlp.helpers import combine_split_result
from itertools import chain
import re
import warnings

def process_into_sentences(doc):
    # remove header
    content = re.split(r"(?i)gives the following[\S\s]*?judgment", doc)
    if len(content) < 2:
        warnings.warn('Not the correct format for judgments')
        content = content[0]
    else:
        content = ''.join(content[1:])

    content = re.sub(r'[^A-Za-z0-9\s.\/-]',r'',content)
    content = re.sub(r'\n',r' ',content)

    sentences = re.split(r"(?<![A-Z])\.", content)

    tokens = [re.findall(r"[\w'\/-]+", sent) for sent in sentences] # find all words

    tokens = [[word for word in sent if len(word) > 1] for sent in tokens] # remove words with one letter only

    # remove numbers (preliminary only; of course Article numbers are important)
    tokens = [[word for word in sent if not word[0].isdigit()] for sent in tokens] 

    # simple word splitting: split words that begin with capitals
    tokens_split = []
    for sent in tokens:
        sent_split = []
        for word in sent:
            # TODO: add splitting exceptions
            s = re.split(r"((?<=[a-z])[A-Z])", word)
            s = combine_split_result(s)
            sent_split.extend(s)
        tokens_split.append(sent_split)
    tokens = tokens_split

    tokens = [sent for sent in tokens if len(sent) > 1] # only deal with longer sentences

    tokens = [[word.lower() for word in sent] for sent in tokens] # lowercase words
    #tokens = [[word for word in sent if word not in self.stopwords] for sent in tokens] # remove stopwords
    return tokens


def process_into_words(doc):
    """Preprocesses content of a given document into
    words instead of sentences. Currently does the same
    thing except for flattening the list afterwards.
    """
    words = list(chain.from_iterable(process_into_sentences(doc)))
    return words