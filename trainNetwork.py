import random

import torch
from torch.autograd import Variable

class TrainNetwork(object):
    def __init__(self, encoder, decoder, output_lang, max_length, teacher_forcing_ratio=0):
        self.encoder = encoder
        self.decoder = decoder
        self.output_lang = output_lang
        self.SOS_token = 0
        self.EOS_token = 1
        self.max_length = max_length
        self.use_cuda = torch.cuda.is_available()
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def train(self, input_variable, target_variable, encoder_optimizer,
              decoder_optimizer, criterion):
        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(self.max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == self.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length

    def evaluate(self, input_variable, sentence):
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(self.max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei],
                                                          encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == self.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        return decoded_words, decoder_attentions[:di + 1]
