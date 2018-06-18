from itertools import chain
from docai.models.word2vec import cosine_similarity
from docai.models.fasttext_training_dataset import FastTextTrainingDataset
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class FastTextSkipgram(nn.Module):
    """Skipgram model for learning fasttext embeddings.
    Assumes that index 0 is the padding index (or equivalently
    the unknown vector).
    Currently FastText is implemented using dense tensors because of
    issues with sparse tensors.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)   
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
        self.embedding_dim = embedding_dim
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def indices2emb(self, indices, embedding='input'):
        """Convert vocabulary indices (from ngrams) to embedding.
        Input params:
        indices: vocabulary indices.
        embedding: 'input' or 'output' embedding. Default: input.
        """
        if embedding not in ['input', 'output']:
            raise ValueError('embedding must be either input or output')

        indices = torch.tensor(indices)
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            indices = indices.cuda()

        if embedding == 'input':
            embeddings = self.u_embeddings(indices)
        else: # if embedding == 'output':
            embeddings = self.v_embeddings(indices)

        return torch.mean(embeddings, dim=0).squeeze().cpu()

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)

        # average subword embeddings
        embed_u = torch.mean(embed_u, dim=1).squeeze()
        embed_v = torch.mean(embed_v, dim=1).squeeze()

        score  = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()
        
        neg_embed_v = self.v_embeddings(v_neg)
        neg_embed_v = torch.mean(neg_embed_v, dim=1).squeeze()
        
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1*loss.sum()/batch_size

class FastText:
    def __init__(self, vocabulary, embedding_dim=200):
        """Initializes the model.
        """
        print('Initializing FastText...')
        self.vocab = vocabulary
        self.embedding_dim = embedding_dim

        # initialize dataset and model 
        self.model = FastTextSkipgram(self.vocab.vocabulary_size, embedding_dim)

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            self.model.cuda()

    def load(self, path):
        """Loads the last model weights.
        """
        folder, _ = os.path.split(path)
        paths = os.listdir(folder)
        paths = [x for x in paths if x.startswith('fasttext_epoch')]
        model_save = torch.load(os.path.join(folder, paths[-1]))
        self.model.load_state_dict(model_save)
    
    def get_embedding_word(self, word, embedding='input'):
        """Return embedding vector for a particular word.
        Input params:
        word: the word we want to embed.
        embedding: either 'input' or 'output'.
        """
        indices = self.vocab.get_indices(word)
        return self.model.indices2emb(indices, embedding).data.numpy()

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

    def train(self, documents, model_save_path, epoch_num=10, batch_size=16, window_size=2,neg_sample_num=10):
        dataset = FastTextTrainingDataset(documents, self.vocab, 
            window_size, batch_size, neg_sample_num)

        optimizer = optim.SGD(self.model.parameters(),lr=0.2)
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

                if batch_num%30000 == 0:
                    torch.save(self.model.state_dict(), model_save_path + '_epoch{}.batch{}.pickle'.format(epoch,batch_num))

                batch_num = batch_num + 1 
        print("Training Finished!")
