from itertools import chain
import math
import numpy as np
import random

class Word2VecTrainingDataset():
    """This class represents the training dataset used to feed pytorch with
    training data.
    """
    def __init__(self, documents, vocab, count, window_size, batch_size, neg_sample_num):
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_num = neg_sample_num

        self.vocab = vocab
        self.count = count
        self.sample_table = self.init_sample_table()

        self.train_data = self.initialize_training_data(documents)
        self.train_data = [sent for sent in self.train_data if len(sent) > 1] # only longer sentences are relevant
        self.train_data = [self.pad(sent, 2*window_size + 1) for sent in self.train_data] # pad short sentences

        self.train_iterator = None
        self.current_sentence = None
        self.sentence_index = 0

    def pad(self, sentence, length):
        if len(sentence) < length:
            pad_length = length - len(sentence)
            pad_begin = [0] * (pad_length // 2)
            pad_end = [0] * (pad_length - len(pad_begin))
            sentence = pad_begin + sentence + pad_end
        return sentence

    def initialize_training_data(self, documents):
        sentences = chain.from_iterable(documents)
        indexed_sentences = []
        for sentence in sentences:
            indexed_words = []
            for word in sentence:
                if word in self.vocab:
                    index = self.vocab[word]
                else:
                    index = 0  # dictionary['UNK']
                indexed_words.append(index)
            indexed_sentences.append(indexed_words)

        train_data = self.subsampling(indexed_sentences)
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

    def subsampling(self, sentences, threshold=1e-5):
        """Perform subsampling according to the word
        count and the frequency probability described
        in the original paper.
        """
        count = [ele[1] for ele in self.count]
        frequency = np.array(count) / sum(count)
        # calculate probability of removal
        P = {idx: ((f-threshold)/f) - math.sqrt(threshold/f) for idx, f in enumerate(frequency)}

        subsampled_sentences = []
        for sentence in sentences:
            subsampled_sentence = []
            for word in sentence:
                if random.random() > P[word]:
                    subsampled_sentence.append(word)
            subsampled_sentences.append(subsampled_sentence)
        return subsampled_sentences

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
        context = np.ndarray(shape=(self.batch_size,2 * self.window_size), dtype=np.int64)
        labels = np.ndarray(shape=(self.batch_size), dtype=np.int64)

        pos_u = []
        pos_v = []

        for i in range(self.batch_size):
            # if we ran out of words, just fetch next sentence
            if (self.sentence_index + span) > len(self.current_sentence):
                self.current_sentence = next(self.train_iterator)
                self.sentence_index = 0

            buffer = self.current_sentence[self.sentence_index:self.sentence_index + span]
            context[i,:] = buffer[:self.window_size]+buffer[self.window_size+1:]
            labels[i] = buffer[self.window_size]

            for j in range(span-1):
                pos_u.append(labels[i])
                pos_v.append(context[i,j])

            self.sentence_index += 1

        neg_v = np.random.choice(self.sample_table, size=(self.batch_size*2*self.window_size, self.neg_sample_num))
        return np.array(pos_u), np.array(pos_v), neg_v