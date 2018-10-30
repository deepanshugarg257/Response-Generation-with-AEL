import random
import torch.nn as nn

import torch
from torch.autograd import Variable

class Train_Network(object):
    def __init__(self, encoder, decoder, discriminator_cnn, discriminator_dense, output_lang, max_length, embedding, input_vocab_size,
                 num_layers=1, batch_size=1):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator_cnn = discriminator_cnn
        self.discriminator_dense = discriminator_dense
        self.output_lang = output_lang
        self.num_layers = num_layers
        self.SOS_token = 1
        self.EOS_token = 2
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = embedding.shape[0]
        self.embedding_length = embedding.shape[1]
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding)
        self.fake_labels = -1*torch.ones(self.batch_size)
        self.true_labels = torch.ones(self.batch_size)
        self.final_criterion = nn.MSELoss()
        self.discriminator_criterion = nn.SoftMarginLoss()
        if self.use_cuda:
            self.embedding = self.embedding.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.true_labels = self.true_labels.cuda()

    def final_train(self, input_variables, target_variables, lengths,
              decoder_optimizer, discriminator_cnn_optimizer, discriminator_dense_optimizer, criterion):

        ''' Pad all tensors in this batch to same length. '''
        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables) # L x B x E
        target_variables = torch.nn.utils.rnn.pad_sequence(target_variables)

        input_length = input_variables.size()[0]
        target_length = target_variables.size()[0]

        query = []
        one_hot = torch.cuda.FloatTensor(self.batch_size, self.input_vocab_size)
        for input in input_variables:
            one_hot.zero_()
            query.append(one_hot.scatter(1, input.view(-1, 1), 1))

        for i in range(input_length, self.max_length):
            if self.use_cuda: query.append(torch.zeros([self.batch_size, self.input_vocab_size]).cuda())
            else: query.append(torch.zeros([self.batch_size, self.input_vocab_size]))

        query = torch.stack(query, 1) # B x max_length x V

        target = []
        one_hot = torch.cuda.FloatTensor(self.batch_size, self.output_vocab_size)
        for t in target_variables:
            one_hot.zero_()
            target.append(one_hot.scatter(1, t.view(-1, 1), 1))

        for i in range(target_length, self.max_length):
            if self.use_cuda: target.append(torch.zeros([self.batch_size, self.output_vocab_size]).cuda())
            else: target.append(torch.zeros([self.batch_size, self.output_vocab_size]))

        target = torch.stack(target, 1) # B x max_length x V

        decoder_optimizer.zero_grad()
        discriminator_cnn_optimizer.zero_grad()
        discriminator_dense_optimizer.zero_grad()
        loss = 0

        encoder_hidden = self.encoder.init_hidden()

        encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)

        decoder_inputs = torch.cuda.FloatTensor(torch.cat([self.embedding(torch.cuda.LongTensor([self.SOS_token]))]*self.batch_size)).unsqueeze(0)
        if self.use_cuda: decoder_inputs = decoder_inputs.cuda()

        decoder_hidden = encoder_hidden[:self.num_layers].view(self.num_layers, self.batch_size, -1)


        ##### training the decoder part
        response = []
        for di in range(target_length):
            decoder_outputs, decoder_hidden = self.decoder(decoder_inputs, decoder_hidden, encoder_outputs, lengths)
            response.append(decoder_outputs)
            decoder_inputs = decoder_outputs.unsqueeze(1) # (B, V) -> (B, 1, V)
            loss += criterion(decoder_outputs, target_variables[di])
            decoder_inputs = torch.mm(decoder_inputs.squeeze(1), self.embedding.weight).unsqueeze(0)  # 1xBxE

        for i in range(target_length, self.max_length):
            if self.use_cuda: response.append(torch.zeros([self.batch_size, self.output_vocab_size]).cuda())
            else: response.append(torch.zeros([self.batch_size, self.output_vocab_size]))

        response = torch.stack(response, 1) # B x max_length x V
        false_representation = self.discriminator_cnn(query, response)
        true_representation = self.discriminator_cnn(query, target)

        decoder_loss = self.final_criterion(false_representation, true_representation)

        false_class = self.discriminator_dense(false_representation)
        true_class = self.discriminator_dense(true_representation)

        decoder_loss.backward(retain_graph = True)
        decoder_optimizer.step()

        discriminator_cnn_optimizer.zero_grad()

        discriminator_loss = 0
        discriminator_loss += self.discriminator_criterion(false_class, self.fake_labels)
        discriminator_loss += self.discriminator_criterion(true_class, self.true_labels)  # -1 = fake, 1 = real
        discriminator_loss.backward()
        discriminator_dense_optimizer.step()
        discriminator_cnn_optimizer.step()

        return loss.item() / target_length, decoder_loss.item()/target_length, discriminator_loss.item()/target_length


    def pre_train(self, input_variables, target_variables, lengths, encoder_optimizer,
              decoder_optimizer, discriminator_cnn_optimizer, discriminator_dense_optimizer, criterion, train_dis):

        ''' Pad all tensors in this batch to same length. '''
        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables) # L x B x E
        target_variables = torch.nn.utils.rnn.pad_sequence(target_variables)

        input_length = input_variables.size()[0]
        target_length = target_variables.size()[0]

        query = []
        one_hot = torch.cuda.FloatTensor(self.batch_size, self.input_vocab_size)
        for input in input_variables:
            one_hot.zero_()
            query.append(one_hot.scatter(1, input.view(-1, 1), 1))

        for i in range(input_length, self.max_length):
            if self.use_cuda: query.append(torch.zeros([self.batch_size, self.input_vocab_size]).cuda())
            else: query.append(torch.zeros([self.batch_size, self.input_vocab_size]))

        query = torch.stack(query, 1) # B x max_length x V

        target = []
        one_hot = torch.cuda.FloatTensor(self.batch_size, self.output_vocab_size)
        for t in target_variables:
            one_hot.zero_()
            target.append(one_hot.scatter(1, t.view(-1, 1), 1))

        for i in range(target_length, self.max_length):
            if self.use_cuda: target.append(torch.zeros([self.batch_size, self.output_vocab_size]).cuda())
            else: target.append(torch.zeros([self.batch_size, self.output_vocab_size]))

        target = torch.stack(target, 1) # B x max_length x V

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        discriminator_cnn_optimizer.zero_grad()
        discriminator_dense_optimizer.zero_grad()
        loss = 0

        encoder_hidden = self.encoder.init_hidden()

        encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)
        decoder_inputs = torch.cuda.FloatTensor(torch.cat([self.embedding(torch.cuda.LongTensor([self.SOS_token]))]*self.batch_size)).unsqueeze(0)
        if self.use_cuda: decoder_inputs = decoder_inputs.cuda()

        decoder_hidden = encoder_hidden[:self.num_layers].view(self.num_layers, self.batch_size, -1)

        response = []
        for di in range(target_length):
            decoder_outputs, decoder_hidden = self.decoder(decoder_inputs, decoder_hidden, encoder_outputs, lengths)
            response.append(decoder_outputs)
            decoder_inputs = decoder_outputs.unsqueeze(1) # (B, V) -> (B, 1, V)
            loss += criterion(decoder_outputs, target_variables[di])
            decoder_inputs = torch.mm(decoder_inputs.squeeze(1), self.embedding.weight).unsqueeze(0)  # 1xBxE

        loss.backward()

        decoder_optimizer.step()
        encoder_optimizer.step()

        for i in range(target_length, self.max_length):
            if self.use_cuda: response.append(torch.zeros([self.batch_size, self.output_vocab_size]).cuda())
            else: response.append(torch.zeros([self.batch_size, self.output_vocab_size]))

        if random.uniform(0, 1) < train_dis:
            response = torch.stack(response, 1).detach()  # B x max_length x V
            false_representation = self.discriminator_cnn(query, response)
            true_representation = self.discriminator_cnn(query, target)
            false_class = self.discriminator_dense(false_representation)
            true_class = self.discriminator_dense(true_representation)

            discriminator_loss = 0
            discriminator_loss += self.discriminator_criterion(false_class, self.fake_labels)
            discriminator_loss += self.discriminator_criterion(true_class, self.true_labels)  # -1 = fake, 1 = real
            discriminator_loss.backward()
            discriminator_dense_optimizer.step()
            discriminator_cnn_optimizer.step()
        # print(false_class, true_class, self.fake_labels, self.true_labels)

        return loss.item() / target_length

    def evaluate(self, input_variable):
        input_variable = torch.nn.utils.rnn.pad_sequence(input_variable)
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden(1)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, [input_length], encoder_hidden)

        decoder_input = self.embedding(torch.cuda.LongTensor([self.SOS_token])).unsqueeze(0)  # SOS
        if self.use_cuda: decoder_input = decoder_input.cuda()

        decoder_hidden = encoder_hidden[:self.num_layers].view(self.num_layers, 1, -1)

        decoded_words = []
        # decoder_attentions = torch.zeros(self.max_length, input_length)

        for di in range(self.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, [input_length])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == self.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.output_lang.index2word[int(ni)])

            decoder_input = decoder_output.unsqueeze(0).permute(1, 0, 2)
            decoder_input = torch.mm(decoder_input.squeeze(1), self.embedding.weight).unsqueeze(0)
            if self.use_cuda: decoder_input = decoder_input.cuda()

        return decoded_words #, decoder_attentions[:di + 1]
