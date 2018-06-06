import collections
from itertools import chain
from nlp.helpers import cosine_similarity
from nlp.word2vec_training_dataset import Word2VecTrainingDataset
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import zipfile

class Skipgram(nn.Module):
    """Skipgram model for learning word2vec embeddings.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
        self.embedding_dim = embedding_dim
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def idx2emb(self, idx):
        idx = torch.tensor(idx)
        return self.u_embeddings(idx)

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)

        score  = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()
        
        neg_embed_v = self.v_embeddings(v_neg)
        
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1*loss.sum()/batch_size

    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

    def save_embedding(self, file_name, id2word):
        embeds = self.u_embeddings.weight.data
        fo = open(file_name, 'w')
        for idx in range(len(embeds)):
            word = id2word(idx)
            embed = ' '.join(embeds[idx])
            fo.write(word+' '+embed+'\n')

class Word2Vec:
    def __init__(self, vocabulary, embedding_dim=200, epoch_num=10, batch_size=16, window_size=2,neg_sample_num=10):
        """Initializes the model by building a vocabulary of most frequent words
        and performing subsamling according to the frequency distribution proposed
        in the word2vec paper.
        """
        print('Initializing Word2Vec...')
        self.vocab = vocabulary
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.window_size = window_size
        self.neg_sample_num = neg_sample_num

        # initialize dataset and model 
        self.model = Skipgram(self.vocab.vocabulary_size, embedding_dim)

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            self.model.cuda()

    def load(self, path):
        """Loads the last model weights and the
        vocabulary that was built.
        """
        folder, _ = os.path.split(path)
        paths = os.listdir(folder)
        paths = [x for x in paths if x.startswith('word2vec_model_epoch')]
        model_save = torch.load(os.path.join(folder, paths[-1]))
        self.model.load_state_dict(model_save)
    
    def get_embedding_word(self, word):
        """Return embedding vector for a particular word.
        """
        emb = self.model.idx2emb(self.vocab.get_index(word)).data.numpy()
        return emb

    def get_embedding_doc(self, doc, strategy='average'):
        """Return embedding vector for a document.
        Input params:
        doc: document for which to get embedding.
        strategy: average word embedding or weight with
        tf-idf weights.
        """
        strategies = ['average', 'tf-idf']
        if strategy not in strategies:
            raise ValueError('strategy has to be either "average" or "tf-idf"')

        doc = list(chain.from_iterable(doc))
        if len(doc) < 1:
            emb = np.zeros((self.embedding_dim))
            return emb

        if strategy == 'average':
            emb = [self.get_embedding_word(w) for w in doc]

        else: # tf-idf
            tf = self.vocab.get_tfidf_weights(doc)
            words = [w if tf.get(w) else 'UNK' for w in doc] # replace unknown words by UNK token
            emb = [tf[w] * self.get_embedding_word(w) for w in words]

        emb = np.mean(emb, axis=0)
        return emb

    def word_similarity(self, word1, word2):
        emb1 = self.get_embedding_word(word1)
        emb2 = self.get_embedding_word(word2)

        return cosine_similarity(emb1, emb2)

    def doc_similarity(self, doc1, doc2, strategy='average'):
        """Compares docs by either calculating an average
        of word vectors in a document or, alternatively,
        weighting the average by tf-idf. Words should be
        organized in documents and not in sentences. This
        function works with lists of words instead of iterators.
        """
        emb1 = self.get_embedding_doc(doc1, strategy)
        emb2 = self.get_embedding_doc(doc2, strategy)

        return cosine_similarity(emb1, emb2)

    def train(self, documents, model_save_path):
        dataset = Word2VecTrainingDataset(documents, self.vocab.get_vocabulary(), 
            self.vocab.get_count(), self.window_size, self.batch_size, self.neg_sample_num)

        optimizer = optim.SGD(self.model.parameters(),lr=0.2)
        for epoch in range(self.epoch_num):
            batch_num = 0

            for pos_u, pos_v, neg_v in dataset:
                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v, self.batch_size)

                if batch_num%1000 == 0:
                    print('Epoch: {0}, Batch: {1}, Loss: {2}'.format(epoch, batch_num, loss.item()))

                loss.backward()
                optimizer.step()

                if batch_num%30000 == 0:
                    torch.save(self.model.state_dict(), model_save_path + '_epoch{}.batch{}.pickle'.format(epoch,batch_num))

                batch_num = batch_num + 1 
        print("Training Finished!")
