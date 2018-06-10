import collections
from docai.nlp.helpers import split_to_ngrams
from itertools import chain
import math
import numpy as np
import pickle
import random

class Vocabulary():
    def __init__(self, vocabulary_size=50000):
        self.vocabulary_size = vocabulary_size

        self.count = []
        self.vocab_words = {}
        self.tfidf = []

        self.data_index = 0 # iteration index

    def initialize_and_save_vocab(self, documents, path):
        """Initializes vocabulary from the sentences iterated by
        documents. 
        """
        words = chain.from_iterable(chain.from_iterable(documents))
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        self.vocab_words = dict()
        for i, word_freq in enumerate(self.count):
            self.vocab_words[word_freq[0]] = i
        unk_count = 0
        words = chain.from_iterable(chain.from_iterable(documents))
        for word in words:
            if word not in self.vocab_words:
                unk_count += 1
        self.count[0][1] = unk_count

        with open(path + '_vocab.pickle', "wb") as f:
            pickle.dump(self.count, f)

    def initialize_and_save_idf(self, documents, path):
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

        with open(path + '_idf.pickle', "wb") as f:
            pickle.dump(self.idf, f)

    def load_idf(self, path):
        """Loads idf table. The vocabulary should already
        be loaded for this to be of any use.
        """
        with open(path + '_idf.pickle', 'rb') as f:
            self.idf = pickle.load(f)

    def load_vocab(self, path):
        """Load vocabulary and perform initialization
        of the sample table.
        """
        with open(path + '_vocab.pickle', 'rb') as f:
            self.count = pickle.load(f)
            self.vocab_words = {vocab_word[0]: idx for idx, vocab_word in enumerate(self.count)}

    def load(self, path):
        """Load everything.
        """
        self.load_vocab(path)
        self.load_idf(path)

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
        """Returns word index or -1 if word is not
        in dictionary.
        """
        idx = self.vocab_words.get(word)
        return -1 if idx is None else idx

    def get_count(self):
        return self.count

    def get_vocabulary(self):
        return self.vocab_words


class FastTextVocabulary(Vocabulary):
    def __init__(self, vocabulary_size=50000):
        super().__init__(vocabulary_size)

    def initialize_and_save_vocab(self, documents, path):
        """Initializes vocabulary from the sentences iterated by
        documents. 
        """
        words = chain.from_iterable(chain.from_iterable(documents))
        word_ngrams = []

        for word in words:
            ngrams = split_to_ngrams(word)
            word_ngrams.extend(ngrams)

        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(word_ngrams).most_common(self.vocabulary_size - 1))
        self.vocab_words = dict()
        for i, word_freq in enumerate(self.count):
            self.vocab_words[word_freq[0]] = i

        unk_count = 0
        words = chain.from_iterable(chain.from_iterable(documents))
        for word in words:
            if word not in self.vocab_words:
                unk_count += 1
        self.count[0][1] = unk_count

        with open(path + '_fasttext_vocab.pickle', "wb") as f:
            pickle.dump(self.count, f)