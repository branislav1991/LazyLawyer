import collections
from docai.nlp.helpers import split_to_ngrams
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
    def __init__(self, vocabulary_size=50000):
        self.vocabulary_size = vocabulary_size

        self.count = []
        self.vocab_words = {}
        self.tfidf = []

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

        with open(path + '.pickle', "wb") as f:
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
        with open(path + '.pickle', 'rb') as f:
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
        """Returns word index or 0 (for UNK token) if word is not
        in dictionary.
        """
        idx = self.vocab_words.get(word)
        return 0 if idx is None else idx

    def get_count(self):
        return self.count


class FastTextVocabulary(Vocabulary):
    def __init__(self, vocabulary_size=50000):
        super().__init__(vocabulary_size)

    def initialize_and_save_vocab(self, documents, path):
        """Initializes vocabulary from the sentences iterated by
        documents. 
        """
        words = chain.from_iterable(chain.from_iterable(documents))
        ngrams = chain.from_iterable((split_to_ngrams(word) for word in words))

        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(ngrams).most_common(self.vocabulary_size - 1))
        self.vocab_words = dict()
        for i, word_freq in enumerate(self.count):
            self.vocab_words[word_freq[0]] = i
        unk_count = 0
        words = chain.from_iterable(chain.from_iterable(documents))
        ngrams = chain.from_iterable((split_to_ngrams(word) for word in words))

        for ngram in ngrams:
            if ngram not in self.vocab_words:
                unk_count += 1
        self.count[0][1] = unk_count

        with open(path + '_fasttext.pickle', "wb") as f:
            pickle.dump(self.count, f)

    def load_vocab(self, path):
        super().load_vocab(path + '_fasttext')

    def initialize_and_save_idf(self, documents, path):
        """We save idf frequency for whole words instead of
        subword ngrams. This makes sense due to the fact that 
        we weight the word vectors after averaging of the subword
        structures.
        """
        super().initialize_and_save_idf(documents, path + '_fasttext')

    def load_idf(self, path):
        super().load_idf(path + '_fasttext')

    def get_indices(self, word):
        """Returns word indices (for all ngrams) or 0 (for UNK token) 
        if word is not in dictionary.
        """
        ngrams = split_to_ngrams(word)
        indices = [self.vocab_words.get(ngram) for ngram in ngrams]
        indices = [0 if idx is None else idx for idx in indices]
        return indices
