from docai import helpers
from docai.nlp.helpers import combine_split_result
from itertools import chain
import re
import warnings
#import Stemmer

#stemmer = Stemmer.Stemmer('english')

def extract_keywords(doc):
    """Extracts keywords which are usually located on top of the judgment.
    """
    keywords = re.search(r'\((.*)(?=\)\s+(?:In Case|In Joined Cases))', doc)
    if not keywords or not keywords.group(1):
        # probably this document comes from a pdf and has no keywords attached
        keywords = None
    else:
        keywords = keywords.group(1)
    return keywords

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
    sentences = re.split(r'(?<!\s[A-Z]|\sp|pp|\.\.|\s\.)\.(?!\.+)', doc)
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

def preprocess(doc_content):
    """Perform all necessary steps to process doc_content from raw string to a
    list of sentences containing word tokens. 
    """
    doc_content = remove_header(doc_content)
    doc_content = normalize(doc_content)
    doc_content = split_to_sentences(doc_content)

    doc_content = [tokenize(sent) for sent in doc_content]
    doc_content = [sent for sent in doc_content if len(sent) > 0] # remove empty phrases
    #doc_content = [stemmer.stemWords(sent) for sent in doc_content] # do not stem
    
    return doc_content
