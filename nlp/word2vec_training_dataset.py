from itertools import chain
import math
import numpy as np
import random

class Word2VecTrainingDataset():
    """This class represents the training dataset used to feed pytorch with
    training data.
    """
    def __init__(self, document_gen, vocab, count, window_size, batch_size, neg_sample_num):
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_num = neg_sample_num

        self.vocab = vocab
        self.count = count
        self.sample_table = self.init_sample_table()

        self.data_index = 0 # iteration index
        self.train_data = self.initialize_training_data(document_gen)

    def initialize_training_data(self, document_gen):
        words = chain.from_iterable(document_gen)
        indexed_words = []
        for word in words:
            if word in self.vocab:
                index = self.vocab[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            indexed_words.append(index)

        train_data = self.subsampling(indexed_words)
        return train_data

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

    def subsampling(self, data, threshold=1e-5):
        """Perform subsampling according to the word
        count and the frequency probability described
        in the original paper.
        """
        count = [ele[1] for ele in self.count]
        frequency = np.array(count) / sum(count)
        # calculate probability of removal
        P = {idx: ((f-threshold)/f) - math.sqrt(threshold/f) for idx, f in enumerate(frequency)}

        subsampled_data = list()
        for word in data:
            if random.random() > P[word]:
                subsampled_data.append(word)
        return subsampled_data

    def reset(self):
        self.data_index = 0

    def __iter__(self):
        return self

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