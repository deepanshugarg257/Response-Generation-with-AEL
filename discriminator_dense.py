import torch
import torch.nn as nn

class Discriminator_Dense(nn.Module):
    def __init__(self, max_length, num_filters=128, dropout_p=0.3):
        super(Discriminator_Dense, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        self.linear = nn.Sequential(
            nn.Dropout(p = dropout_p),
            nn.Linear(2*num_filters*(int(max_length/2)+1), 1)
        )
        self.tanh = nn.Tanh()

        if self.use_cuda:
            self.linear = self.linear.cuda()
            self.tanh = self.tanh.cuda()

    def forward(self, concatenated):
        op = self.tanh(self.linear(concatenated)).squeeze(1)
        return op