import torch
import torch.nn as nn

class Discriminator_CNN(nn.Module):
    def __init__(self, query_vocab_size, response_vocab_size,
                 window_size = 2, num_filters=128):
        super(Discriminator_CNN, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        self.response_cnn = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=[window_size, response_vocab_size], padding = [1, 0]),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.query_cnn = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=[window_size, query_vocab_size], padding = [1, 0]),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=[window_size,1], padding = [1, 0])

        if self.use_cuda:
            self.response_cnn = self.response_cnn.cuda()
            self.query_cnn = self.query_cnn.cuda()
            self.max_pool = self.max_pool.cuda()

    def forward(self, query, response):
        batch_size = response.size()[0]
        response = response.unsqueeze(1) # B x 1 x len x vocab
        query = query.unsqueeze(1) # B x 1 x len x vocab

        response_representation = self.max_pool(self.response_cnn(response)).view(batch_size, -1)
        query_representation = self.max_pool(self.query_cnn(query)).view(batch_size, -1)
        concatenated = torch.cat([response_representation, query_representation], 1)

        return concatenated