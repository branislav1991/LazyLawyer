import helpers
import re

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

    def __next__(self):
        content = next(self.content_generator)

        print('Preprocessing document...')
        # remove header
        content = re.split(r"(?i)gives the following[\S\s]*?judgment", content)
        if len(content) < 2:
            raise ValueError('Not the correct format for judgments')
        
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
                sent_split.extend(re.split(r"(?<=[a-z])[A-Z]", word))
            tokens_split.append(sent_split)
        tokens = tokens_split

        tokens = [sent for sent in tokens if len(sent) > 1] # only deal with longer sentences

        tokens = [[word.lower() for word in sent] for sent in tokens] # lowercase words
        tokens = [[word for word in sent if word not in self.stopwords] for sent in tokens] # remove stopwords

        return tokens

    def __iter__(self):
        return self