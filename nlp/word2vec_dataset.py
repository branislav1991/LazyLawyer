import collections
import itertools
import numpy as np

import math
import os
import pickle
import random

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

class Word2VecDataset():
    def __init__(self, vocabulary_size, window_size, batch_size, neg_sample_num):
        """Takes a sentences generator as input as well as
        the desired vocabulary size.
        """
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_num = neg_sample_num
        self.vocabulary_size = vocabulary_size

        self.data_index = 0 # iteration index

    def initialize_vocab(self, sentences):
        self.words = list(itertools.chain.from_iterable(sentences))

        indexed_words, self.count, self.vocab_words = self.build_dataset(self.words,
                                                                self.vocabulary_size) 
        self.train_data = self.subsampling(indexed_words)
        self.sample_table = self.init_sample_table()

    def load_vocab(self, path):
        pass

    def build_dataset(self, words, n_words):
        """Process word inputs into an index and build
        vocabulary.
        """
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for i, word_freq in enumerate(count):
            dictionary[word_freq[0]] = i
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        return data, count, dictionary

    def get_index(self, word):
        """Returns word index or -1 if word is not
        in dictionary.
        """
        idx = self.vocab_words.get(word)
        return 0 if idx is None else idx

    def load_vocab(self, path):
        """Load vocabulary and perform initialization
        of the sample table.
        """
        with open(path + '_vocab.pickle', 'rb') as f:
            self.count = pickle.load(f)
            self.vocab_words = {vocab_word[0]: idx for idx, vocab_word in enumerate(self.count)}

    def save_vocab(self, path):
        """Save vocabulary (before subsampling and initializing of
        the sample table). The model path has to be provided
        without extension.
        """
        with open(path + '_vocab.pickle', "wb") as f:
            pickle.dump(self.count, f)
            # for vocab_word, idx in self.vocab_words.items():
            #     f.write("%s %d\n" % (vocab_word, self.count[idx][1])) 

    def init_sample_table(self, table_size=1e6):
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency/ power
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def subsampling(self, data, threshold=1e-5):
        count = [ele[1] for ele in self.count]
        frequency = np.array(count) / sum(count)
        # calculate probability of removal
        P = {idx: ((f-threshold)/f) - math.sqrt(threshold/f) for idx, f in enumerate(frequency)}

        subsampled_data = list()
        for word in data:
            if random.random() > P[word]:
                subsampled_data.append(word)
        return subsampled_data

    def __next__(self):
        data = self.train_data
        span = 2 * self.window_size + 1
        context = np.ndarray(shape=(self.batch_size,2 * self.window_size), dtype=np.int64)
        labels = np.ndarray(shape=(self.batch_size), dtype=np.int64)
        pos_pair = []

        if self.data_index + span > len(data):
            raise StopIteration()

        buffer = data[self.data_index:self.data_index + span]
        pos_u = []
        pos_v = []

        for i in range(self.batch_size):
            self.data_index += 1
            context[i,:] = buffer[:self.window_size]+buffer[self.window_size+1:]
            labels[i] = buffer[self.window_size]
            if self.data_index + span > len(data):
                temp_index = self.data_index + span - 1 - len(data)
                buffer[:] = data[temp_index:temp_index + span]

            else:
                buffer = data[self.data_index:self.data_index + span]

            for j in range(span-1):
                pos_u.append(labels[i])
                pos_v.append(context[i,j])
        neg_v = np.random.choice(self.sample_table, size=(self.batch_size*2*self.window_size, self.neg_sample_num))
        return np.array(pos_u), np.array(pos_v), neg_v

    def __iter__(self):
        return self

    def reset(self):
        """If dataset is already exhausted, allows
        resetting the pointer and looping through the
        dataset again.
        """
        self.data_index = 0