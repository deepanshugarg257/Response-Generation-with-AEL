import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from attention import Attention

class Decoder_RNN(nn.Module):
    def __init__(self, hidden_size, embedding, num_layers=1, use_embedding=False,
                 train_embedding=True, dropout_p=0.1):
        super(Decoder_RNN, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.root_var = 0.1**0.5

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # Size of embedding vector
            self.output_size = embedding.shape[0] # Number of words in vocabulary
            self.embedding_mat = self.embedding.weight.unsqueeze(0)

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1] # Size of embedding vector
            self.output_size = embedding[0] # Number of words in vocabulary

        self.embedding.weight.requires_grad = train_embedding

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, input_lengths):
        '''
        input           -> (Batch Size x 1 x Vocab. len.)
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        encoder_outputs -> (Max Sentence Length, Batch Size, Hidden Size * Num. Directions)
        input_lengths   -> (Batch Size (Sorted in decreasing order of lengths))
        '''
        batch_size = input.size()[0]

        output, hidden = self.gru(input, hidden)

        output = output.squeeze(0) # (1, B, V) -> (B, V)
        z = torch.randn(batch_size, output.size()[1]) * self.root_var # to make variance 0.1

        if self.use_cuda:
            z = z.cuda()

        output = F.log_softmax(self.out(output + z), dim=1)
        return output, hidden #, attn_weights

    def init_hidden(self, batch_size):
        result = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    # def final_train(self):
