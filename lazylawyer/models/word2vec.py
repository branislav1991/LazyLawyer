from itertools import chain
from lazylawyer.nlp.helpers import cosine_similarity
from lazylawyer.models.word2vec_training_dataset import Word2VecTrainingDataset
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class Skipgram(nn.Module):
    """Skipgram model for learning word2vec embeddings.
    Assumes that index 0 is the padding index (or equivalently
    the unknown vector).
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)   
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
        self.embedding_dim = embedding_dim
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def idx2emb(self, idx, embedding='input'):
        """Convert vocabulary index to embedding.
        Input params:
        idx: vocabulary index.
        embedding: 'input' or 'output' embedding. Default: input.
        """
        if embedding not in ['input', 'output']:
            raise ValueError('embedding must be either input or output')

        idx = torch.tensor(idx)
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            idx = idx.cuda()

        if embedding == 'input':
            return self.u_embeddings(idx).cpu()
        else: # if embedding == 'output':
            return self.v_embeddings(idx).cpu()

    def weights_from_pretrained(self, idx_dict):
        """Load pretrained weights from a dictionary of
        word indices and embedding vectors.
        """
        for idx in idx_dict.keys():
            self.u_embeddings.weight[idx,:] = torch.FloatTensor(idx_dict[idx])

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

class Word2Vec:
    def __init__(self, vocabulary, embedding_dim):
        """Initializes the model. 
        """
        print('Initializing Word2Vec...')
        self.vocab = vocabulary
        self.embedding_dim = embedding_dim
        self.model = Skipgram(self.vocab.vocab_size(), self.embedding_dim)

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            self.model.cuda()

    def load_from_dict(self, idx_dict):
        """Loads a pretrained model from a word dictionary.
        """
        self.model.weights_from_pretrained(idx_dict)

    def load(self, path):
        """Loads the last model weights.
        """
        folder, name = os.path.split(path)
        paths = os.listdir(folder)
        paths = [x for x in paths if x.startswith(name + '_epoch') or x.startswith(name + '_final')]
        model_save = torch.load(os.path.join(folder, paths[-1]))
        self.model.load_state_dict(model_save)
    
    def get_embedding_word(self, word, embedding='input'):
        """Return embedding vector for a particular word.
        Input params:
        word: the word that we want to embed.
        embedding: either 'input' or 'output'.
        """
        idx = self.vocab.get_index(word)
        return self.model.idx2emb(idx, embedding).data.numpy()

    def get_embedding_doc(self, doc, strategy='average', embedding='input'):
        """Return embedding vector for a document.
        Input params:
        doc: document for which to get embedding.
        strategy: average word embedding or weight with
        tf-idf weights.
        embedding: either 'input' or 'output'.
        """
        strategies = ['average', 'tf-idf']
        if strategy not in strategies:
            raise ValueError('strategy has to be either "average" or "tf-idf"')

        doc = list(chain.from_iterable(doc))
        if len(doc) < 1:
            emb = np.zeros((self.embedding_dim))
            return emb

        if strategy == 'average':
            emb = [self.get_embedding_word(w, embedding) for w in doc]

        else: # tf-idf
            tf = self.vocab.get_tfidf_weights(doc)
            words = [w if tf.get(w) else 'UNK' for w in doc] # replace unknown words by UNK token
            emb = [tf[w] * self.get_embedding_word(w, embedding) for w in words]

        emb = np.mean(emb, axis=0)
        return emb

    def save(self, model_save_filename):
        """Save current model state.
        """
        torch.save(self.model.state_dict(), model_save_filename)

    def train(self, documents, model_save_path, epoch_num, batch_size, window_size, neg_sample_num, learning_rate):
        dataset = Word2VecTrainingDataset(documents, self.vocab, 
            window_size, batch_size, neg_sample_num)

        optimizer = optim.SGD(self.model.parameters(),lr=learning_rate)
        for epoch in range(epoch_num):
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
                loss = self.model(pos_u, pos_v, neg_v, batch_size)

                if batch_num%1000 == 0:
                    print('Epoch: {0}, Batch: {1}, Loss: {2}'.format(epoch, batch_num, loss.item()))

                loss.backward()
                optimizer.step()

                if batch_num%300000 == 0:
                    torch.save(self.model.state_dict(), model_save_path + '_epoch{}.batch{}.pickle'.format(epoch,batch_num))

                batch_num = batch_num + 1 
        print("Training Finished!")
