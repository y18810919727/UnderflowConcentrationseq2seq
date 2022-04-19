#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1, max_length=80):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size + input_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[-1]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(1, 0))

        output = torch.cat((embedded[0], attn_applied[:, 0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.tanh(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttentionSeq2Seq(torch.nn.Module):
    def __init__(self, k_in, k_out, k_state, max_length=80, num_layers=1):

        """
        :param k_in: size of external input
        :param k_out: size of system output
        :param k_state: size of hidden state
        :param encoder_net_type: The type of sequential encoder network, RNN or LSTM or GRU.
        """
        super(AttentionSeq2Seq, self).__init__()
        self.max_length = max_length
        self.encoder = EncoderRNN(k_in + k_out, k_state, num_layers)
        self.decoder = AttnDecoderRNN(k_in, k_state, k_out, num_layers)

    def forward(self, input):

        pre_x, pre_y, forward_x = input
        device = forward_x.device

        if pre_x.shape[0] == 0:
            pre_x = torch.zeros((1, pre_x.shape[1], pre_x.shape[2])).to(forward_x.device)
            pre_y = torch.zeros((1, pre_y.shape[1], pre_y.shape[2])).to(forward_x.device)

        forward_length, batch_size, _ = forward_x.shape
        # import pdb
        # pdb.set_trace()
        assert pre_y.size()[0] <= self.max_length
        encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=device)

        encoder_hidden = self.encoder.init_hidden(batch_size, device)
        encoder_outputs_short, encoder_hidden = self.encoder(torch.cat([pre_x, pre_y], dim=2), encoder_hidden)
        encoder_outputs[:encoder_outputs_short.size()[0]] = encoder_outputs_short

        decoder_output = torch.zeros_like(pre_y[0])
        decoder_hidden = encoder_hidden
        estimate_y_list = []
        for di in range(forward_length):
            decoder_input = torch.cat((decoder_output, forward_x[di]), dim=-1).unsqueeze(0)
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )
            estimate_y_list.append(decoder_output)
        estimate_y_all = torch.stack(estimate_y_list, dim=0)
        return estimate_y_all


