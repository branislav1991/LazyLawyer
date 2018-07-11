from lazylawyer.nlp.helpers import split_to_ngrams
from itertools import chain
import math
import numpy as np
import random

class FastTextTrainingDataset():
    """This class represents the training dataset used to feed pytorch with
    training data for the fasttext algorithm.
    """
    def __init__(self, documents, vocab, window_size, batch_size, neg_sample_num):
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_num = neg_sample_num

        self.vocab = vocab
        self.count = vocab.get_count()
        self.sample_table = self.init_sample_table()

        # word containing lagest number of ngrams for padding
        # if a word contains more ngrams, they will be cut
        self.max_ngram = self.vocab.max_ngram 

        self.train_data = self.initialize_training_data(documents)
        self.train_data = [sent for sent in self.train_data if len(sent) > 1] # only longer sentences are relevant
        self.train_data = [self.pad_sentence(sent, 2*window_size + 1, self.max_ngram) for sent in self.train_data]

        self.train_iterator = None
        self.current_sentence = None
        self.sentence_index = 0

    def initialize_training_data(self, documents, subsampling_threshold=1e-4):
        sentences = chain.from_iterable(documents)

        indexed_sentences = []
        for sentence in sentences:
            indexed_sentence = []
            for word in sentence:
                indexed_ngrams = []
                ngrams = split_to_ngrams(word, max_ngram=self.max_ngram)
                ngrams = self.pad_ngrams(ngrams, self.max_ngram)
                
                for ngram in ngrams:
                    indexed_ngrams.append(self.vocab.get_index(ngram))
                indexed_sentence.append(indexed_ngrams)
            indexed_sentences.append(indexed_sentence)

        train_data = self.subsampling(indexed_sentences, subsampling_threshold)
        return train_data

    def pad_ngrams(self, ngrams, length):
        if len(ngrams) < length:
            pad_length = length - len(ngrams)
            pad = [0] * pad_length
            ngrams = ngrams + pad
        return ngrams

    def pad_sentence(self, sentence, length, max_ngram):
        """Pad sentence with UNK tokens.
        """
        if len(sentence) < length:
            pad_length = length - len(sentence)
            pad = [[0] * max_ngram] * pad_length
            sentence = sentence + pad
        return sentence

    def init_sample_table(self, table_size=1e6):
        """Initializes table to sample from while training.
        Only use this function if you want to train. It is
        not required for predicting.
        """
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency/ power
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def subsampling(self, sentences, threshold):
        """Perform subsampling according to the word
        count and the frequency probability described
        in the original paper. Note that the subsampling
        is only performed according to the whole word
        without taking n-grams into account.
        """
        count = [ele[1] for ele in self.count]
        frequency = np.array(count) / sum(count)
        epsilon = 10e-6
        # calculate probability of removal
        P = {idx: ((f-threshold)/(f+epsilon)) - math.sqrt(threshold/(f+epsilon)) for idx, f in enumerate(frequency)}
        P[0] = 0.5 # set probability of removal of UNK token to 0.5

        subsampled_sentences = []
        for sentence in sentences:
            subsampled_sentence = []
            for ngrams in sentence:
                word = ngrams[0] # first ngram is the actual word
                if random.random() > P[word]:
                    subsampled_sentence.append(ngrams)
            subsampled_sentences.append(subsampled_sentence)
        return subsampled_sentences

    def _index_sentence(self, sentence):
        """Obtains indices of ngrams of the words contained in
        the sentence.
        """
        indexed_sentence = []
        for word in sentence:
            indexed_ngrams = []
            ngrams = split_to_ngrams(word, max_ngram=self.max_ngram)
            ngrams = self.pad_ngrams(ngrams, self.max_ngram)
            
            for ngram in ngrams:
                indexed_ngrams.append(self.vocab.get_index(ngram))
            indexed_sentence.append(indexed_ngrams)
        return indexed_sentence

    def __iter__(self):
        """Resets all iterations and starts a new iteration
        through the dataset.
        """
        self.train_iterator = iter(self.train_data)
        self.current_sentence = next(self.train_iterator)
        self.sentence_index = 0
        return self

    def __next__(self):
        span = 2 * self.window_size + 1
        context = np.ndarray(shape=(self.batch_size, 2 * self.window_size, self.max_ngram), dtype=np.int64)
        labels = np.ndarray(shape=(self.batch_size, self.max_ngram), dtype=np.int64)

        pos_u = []
        pos_v = []

        for i in range(self.batch_size):
            # if we ran out of words, just fetch next sentence
            if (self.sentence_index + span) > len(self.current_sentence):
                self.current_sentence = next(self.train_iterator)
                self.sentence_index = 0

            buffer = self.current_sentence[self.sentence_index:self.sentence_index + span]
            context[i,:,:] = buffer[:self.window_size]+buffer[self.window_size+1:]
            labels[i,:] = buffer[self.window_size]

            for j in range(span-1):
                pos_u.append(labels[i])
                pos_v.append(context[i,j])

            self.sentence_index += 1

        neg_v = np.random.choice(self.sample_table, size=(self.batch_size*2*self.window_size, self.max_ngram, self.neg_sample_num))
        return np.array(pos_u), np.array(pos_v), neg_v
