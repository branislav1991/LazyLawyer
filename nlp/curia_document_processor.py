import helpers
from nlp.helpers import combine_split_result
import re
import warnings

class CURIADocumentProcessor():
    """This class processes content one-by-one into lists
    of tokens by DOCUMENTS by applying
    various preprocessing steps such as tokenization etc.
    It requires either a content generator or a list of
    documents as input.
    """
    def __init__(self, content_generator):
        if isinstance(content_generator, list): # if we supplied a list, make it an iterator
            self.content_generator = iter(content_generator)
        else:
            self.content_generator = content_generator

        with open(helpers.setup_json['stopwords_path'], 'r') as stopwords:
            self.stopwords = set(stopwords.read().split(','))

    def preprocess(self, content):
        """Preprocesses content of a given document.
        """
        # remove header
        content = re.split(r"(?i)gives the following[\S\s]*?judgment", content)
        if len(content) < 2:
            warnings.warn('Not the correct format for judgments')
            content = content[0]
        else:
            content = ''.join(content[1:])

        content = re.sub(r'[^A-Za-z0-9\s.\/-]',r'',content)
        content = re.sub(r'\n',r' ',content)

        #sentences = re.split(r"(?<![A-Z])\.", content)

        tokens = re.findall(r"[\w'\/-]+", content) # find all words

        tokens = [word for word in tokens if len(word) > 1] # remove words with one letter only

        # remove numbers (preliminary only; of course Article numbers are important)
        tokens = [word for word in tokens if not word[0].isdigit()] 

        # simple word splitting: split words that begin with capitals
        tokens_split = []
        for word in tokens:
            # TODO: add splitting exceptions
            s = re.split(r"((?<=[a-z])[A-Z])", word)
            s = combine_split_result(s)
            tokens_split.extend(s)
        tokens = tokens_split

        tokens = [word.lower() for word in tokens] # lowercase words
        #tokens = [word for word in tokens if word not in self.stopwords] # remove stopwords
        return tokens

    def __next__(self):
        content = next(self.content_generator)
        return self.preprocess(content)

    def __iter__(self):
        return self