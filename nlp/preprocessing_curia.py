import helpers
from nlp.helpers import combine_split_result
import re
import warnings

class CURIAContentProcessor():
    """This class processes content one-by-one by applying
    various preprocessing steps such as tokenization etc.
    It requires a content generator as input.
    """
    def __init__(self, content_generator):
        self.content_generator = content_generator
        self.buffered_sents = [] # buffer for processed sentences

        with open(helpers.setup_json['stopwords_path'], 'r') as stopwords:
            self.stopwords = set(stopwords.read().split(','))

    def preprocess_and_buffer(self, content):
        """Preprocesses content of a given document and
        adds the preprocessed tokenized sentences to the
        buffered_sents list.
        """
        # remove header
        content = re.split(r"(?i)gives the following[\S\s]*?judgment", content)
        if len(content) < 2:
            warnings.warn('Not the correct format for judgments')
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
        self.buffered_sents.extend(tokens)

    def __next__(self):
        if len(self.buffered_sents) < 1:
            content = next(self.content_generator)
            self.preprocess_and_buffer(content)

        return self.buffered_sents.pop(0)

    def __iter__(self):
        return self