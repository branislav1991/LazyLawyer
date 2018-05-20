import collections
from nlp.word2vec_dataset import Word2VecDataset
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import zipfile

class Skipgram(nn.Module):
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
    def __init__(self, sentences, path, vocabulary_size=100000, embedding_dim=200, epoch_num=10, batch_size=16, windows_size=5,neg_sample_num=10):
        """Initializes the model by building a vocabulary of most frequent words
        and performing subsamling according to the frequency distribution proposed
        in the word2vec paper.
        """
        print('Initializing Word2Vec and building vocabulary...')
        self.path = path
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        
        self.dataset = Word2VecDataset(sentences, vocabulary_size, windows_size, batch_size, neg_sample_num)

    def save_vocab(self):
        self.dataset.save_vocab(self.path)

    def train(self):
        print('Starting training...')
        model = Skipgram(self.vocabulary_size, self.embedding_dim)

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            model.cuda()

        optimizer = optim.SGD(model.parameters(),lr=0.2)
        for epoch in range(self.epoch_num):
            batch_num = 0
            self.dataset.reset()

            for pos_u, pos_v, neg_v in self.dataset:
                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, self.batch_size)

                if batch_num%1000 == 0:
                    print('Epoch: {0}, Batch: {1}, Loss: {2}'.format(epoch, batch_num, loss.item()))

                loss.backward()
                optimizer.step()

                if batch_num%30000 == 0:
                    torch.save(model.state_dict(), self.path + '_epoch{}.batch{}'.format(epoch,batch_num))

                batch_num = batch_num + 1 
        print("Training Finished!")



# Useful functions to use with word2vec
def cosine_similarity(v1,v2):
    """Compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||).
    """
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)