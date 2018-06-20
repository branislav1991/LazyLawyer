from itertools import chain
import math
import numpy as np
import random

class ELMoTrainingDataset():
    """This class represents the training dataset used to feed pytorch with
    training data.
    """
    def __init__(self, documents, vocab, batch_size, max_length=50):
        self.batch_size = batch_size

        self.vocab = vocab
        self.count = vocab.get_count()

        self.train_data = self.initialize_training_data(documents)
        self.train_data = [sent for sent in self.train_data if len(sent) > 1] # only longer sentences are relevant
        self.train_data = [self.trim(sent, max_length) for sent in self.train_data]

        self.train_iterator = None
        self.current_sentence = None
        self.sentence_index = 0

    def trim(self, sentence, length):
        """Trims or pads the sentence to the specified
        length.
        """
        if len(sentence) < length:
            pad_length = length - len(sentence)
            pad = [0] * pad_length
            sentence = sentence + pad
        elif len(sentence) > length:
            sentence = sentence[:length]
        return sentence

    def initialize_training_data(self, documents, subsampling_threshold=1e-4):
        sentences = chain.from_iterable(documents)
        indexed_sentences = []
        for sentence in sentences:
            indexed_words = []
            for word in sentence:
                index = self.vocab.get_index(word)
                indexed_words.append(index)
            indexed_sentences.append(indexed_words)

        return indexed_sentences

    def __iter__(self):
        """Resets all iterations and starts a new iteration
        through the dataset.
        """
        self.train_iterator = iter(self.train_data)
        return self

    def __next__(self):
        next_sentence = next(self.train_iterator)
        sentences = [next_sentence]
        for i in range(1, self.batch_size):
            try:
                sentences.append(next(self.train_iterator))
            except StopIteration:
                break

        return sentences
