from docai.models.elmo_training_dataset import ELMoTrainingDataset
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ELMoModule(nn.Module):
    """ELMo model for learning embeddings with biLSTM.
    """
    def __init__(self, vocab_dim, embedding_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim

        self.embeddings = nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_dim)

    def forward(self, batch):
        batch_size = batch.shape[0]
        hidden_init = torch.zeros(1, batch_size, self.hidden_dim)
        cell_init = torch.zeros(1, batch_size, self.hidden_dim)

        embeds = self.embeddings(batch)
        lstm_out, _ = self.LSTM(embeds, (hidden_init, cell_init))
        linear_out = self.linear(lstm_out)
        scores = F.log_softmax(linear_out, dim=2)
        loss = F.nll_loss(scores.view(-1, self.vocab_dim, batch_size), batch.view(-1, batch_size))
        return loss

    def save_embedding(self):
        pass

class ELMo:
    def __init__(self, vocabulary, embedding_dim=256, hidden_dim=256):
        pass
        """Initializes the model.
        """
        print('Initializing ELMo...')
        self.vocab = vocabulary
        self.embedding_dim = embedding_dim

        # initialize dataset and model 
        self.model = ELMoModule(self.vocab.vocabulary_size, embedding_dim, hidden_dim)

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
            self.model.cuda()

    def load(self, path):
        """Loads the last model weights.
        """
        folder, _ = os.path.split(path)
        paths = os.listdir(folder)
        paths = [x for x in paths if x.startswith('elmo_epoch')]
        model_save = torch.load(os.path.join(folder, paths[-1]))
        self.model.load_state_dict(model_save)

    def train(self, documents, model_save_path, epoch_num=10, batch_size=16):
        dataset = ELMoTrainingDataset(documents, self.vocab, batch_size)

        optimizer = optim.SGD(self.model.parameters(),lr=1)
        for epoch in range(epoch_num):
            batch_num = 0

            for batch in dataset:
                # PyTorch expects input of shape (batch, seq_len, input_size)
                # Output: (batch, seq_len, hidden_size * num_directions)
                batch = Variable(torch.LongTensor(batch))

                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] > 4:
                    batch = batch.cuda()

                optimizer.zero_grad()
                loss = self.model(batch)

                if batch_num%10 == 0:
                    print('Epoch: {0}, Batch: {1}, Loss: {2}'.format(epoch, batch_num, loss.item()))

                loss.backward()
                optimizer.step()

                if batch_num%30000 == 0:
                    torch.save(self.model.state_dict(), model_save_path + '_epoch{}.batch{}.pickle'.format(epoch,batch_num))

                batch_num = batch_num + 1 
        print("Training Finished!")
