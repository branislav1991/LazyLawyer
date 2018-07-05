import collections
from docai.nlp.helpers import split_to_ngrams, add_special_tags
from itertools import chain
import math
import numpy as np
import pickle
import random

class Vocabulary():
    """Represents a vocabulary of words in a corpus. The words
    are indexed and you can obtain the index by calling
    Vocabulary.get_index(). Index is 0 for unknown words.
    """
    def __init__(self, **kwargs):
        """Initialize vocabulary.
        """
        if 'count' in kwargs: # user wants to load existing vocab
            self.load(kwargs['count'])
        elif 'documents' in kwargs: # generate a new vocab out of documents and save to path
            self.initialize_vocab(kwargs['documents'], kwargs['max_vocab_size'])
        elif 'vocab_dict' in kwargs:
            self.load_from_dict(kwargs['vocab_dict'])
        else:
            raise ValueError('Wrong keyword arguments for vocabulary initialization')

    def initialize_vocab(self, documents, max_vocab_size):
        """Initializes vocabulary from the sentences iterated by
        documents. 
        """
        words = chain.from_iterable(chain.from_iterable(documents))
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(max_vocab_size - 1))
        self.vocab_words = dict()
        for i, word_freq in enumerate(self.count):
            self.vocab_words[word_freq[0]] = i
        unk_count = 0
        words = chain.from_iterable(chain.from_iterable(documents))
        for word in words:
            if word not in self.vocab_words:
                unk_count += 1
        self.count[0][1] = unk_count

    def initialize_idf(self, documents):
        """Generates a idf table for every word in the
        vocabulary by iterating over the document iterator
        'documents'. Requires the vocabulary to be loaded.
        """
        num_docs = 0
        doc_freq = [[c[0], 0] for c in self.count]

        for doc in documents:
            words = chain.from_iterable(doc)
            num_docs = num_docs + 1
            for i, entry in enumerate(doc_freq):
                if i > 0 and entry[0] in words: # skip unknown token
                    doc_freq[i][1] = doc_freq[i][1] + 1

        self.idf = {e[0]: math.log((num_docs+1) / (e[1]+1)) for e in doc_freq}
        self.idf['UNK'] = 0.0 # unknown token has idf of 0

    def vocab_size(self):
        return len(self.count)

    def load(self, count):
        """Load vocabulary and perform initialization
        of the sample table.
        """
        self.count = count
        self.vocab_words = {vocab_word[0]: idx for idx, vocab_word in enumerate(self.count)}

    def load_idf(self, idf):
        self.idf = idf

    def load_from_dict(self, word_dict):
        """Load vocabulary from the dictionary of
        words and embeddings.
        """
        idx_dict = {0: np.zeros(300)}
        self.vocab_words['UNK'] = 0
        for i, word in enumerate(word_dict.items()):
            self.vocab_words[word[0]] = i+1
            idx_dict[i+1] = word[1]

        # we do not have any idf info, so just assign the same weight to all
        self.idf = {word[0]: 1.0 for word in word_dict.items()}
        self.idf['UNK'] = 0.0

        self.count = [[word, 1] for word in self.idf.keys()]

        return idx_dict

    def get_tfidf_weights(self, doc):
        """Get term frequency of a word in a document.
        """
        tf = {c[0]: 0 for c in self.count}
        for word in doc:
            if word in tf:
                tf[word] = tf[word] + 1
        tf = {key: value / len(doc) for key, value in tf.items()}

        tfidf = {word: tf[word] * self.get_idf_weight(word) for word in tf.keys() & self.idf.keys()}
        return tfidf

    def get_idf_weight(self, word):
        """Get idf weight for a word or None if word
        not in dictionary.
        """
        return self.idf.get(word)

    def get_index(self, word):
        """Returns word index or 0 (for UNK token) if word is not
        in dictionary.
        """
        idx = self.vocab_words.get(word)
        return 0 if idx is None else idx

    def get_count(self):
        return self.count

    def get_idf(self):
        return self.idf


class FastTextVocabulary(Vocabulary):
    def __init__(self, **kwargs):
        """Initialize vocabulary. There are two options for this:
        Either initialize a vocabulary from documents by passing
        max_vocab_size, max_ngram_size, max_ngram, documents and 
        vocabulary path arguments, or by passing a saved vocabulary 
        to load.
        """
        if 'count' not in kwargs.keys():
            self.max_ngram_size = kwargs['max_ngram_size']
        elif 'vocab_dict' in kwargs:
            raise ValueError('Pretrained embeddings for fasttext not implemented')

        self.max_ngram = kwargs['max_ngram']
        super().__init__(**kwargs)

    def initialize_vocab(self, documents, max_vocab_size):
        """Initializes vocabulary from the sentences iterated by
        documents. 
        """
        words = chain.from_iterable(chain.from_iterable(documents))
        words = (add_special_tags(w) for w in words)

        self.count = [['UNK', -1]]

        # first, add words to the vocabulary, then add ngrams
        self.count.extend(collections.Counter(words).most_common(max_vocab_size - 1))

        words = chain.from_iterable(chain.from_iterable(documents))
        ngrams = chain.from_iterable((split_to_ngrams(word, max_ngram=self.max_ngram)[1:] for word in words))
        self.count.extend(collections.Counter(ngrams).most_common(self.max_ngram_size - 1))

        self.vocab_words = dict()
        for i, word_freq in enumerate(self.count):
            self.vocab_words[word_freq[0]] = i
        unk_count = 0

        words = chain.from_iterable(chain.from_iterable(documents))

        for word in words:
            if word not in self.vocab_words:
                unk_count += 1
        self.count[0][1] = unk_count

    def initialize_idf(self, documents):
        raise NotImplementedError('tf-idf weights for fasttext documents are not supported')

    def load_idf(self, idf):
        raise NotImplementedError('tf-idf weights for fasttext documents are not supported')

    def get_indices(self, word):
        """Returns word indices (for all ngrams) or 0 (for UNK token) 
        if word is not in dictionary.
        """
        ngrams = split_to_ngrams(word, self.max_ngram)
        indices = [self.vocab_words.get(ngram) for ngram in ngrams]
        indices = [0 if idx is None else idx for idx in indices]
        return indices
