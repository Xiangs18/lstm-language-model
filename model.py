import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import const
import math

class LanguageModel(nn.Module):

    def __init__(self, voc_size, dim_word, dim_rnn, num_layers, dropout_rate = 0.2):
        
        super(LanguageModel, self).__init__()
        # Hyperparamters
        self.rnn_layers = num_layers
        self.dim_rnn = dim_rnn
        
        # Layers
        self.dropout = nn.Dropout(dropout_rate)
        self.word_lut = nn.Embedding(voc_size + 1, dim_word, padding_idx=const.PAD)
        self.lstm = nn.LSTM(dim_word, dim_rnn, num_layers, dropout=dropout_rate)
        self.linear_output = nn.Linear(dim_rnn, voc_size + 1)
        self.logprob = nn.LogSoftmax()
        
        # Model train status
        self.train_info = {}
        self.train_info['val loss'] = 100
        self.train_info['train loss'] = 100
        self.train_info['epoch idx'] = 1
        self.train_info['batch idx'] = 1
        self.train_info['val ppl'] = math.exp(100)
        
        # Dictionary for token to index
        self.dictionary = None


    def forward(self, inputs, lengths):
        lengths = lengths.contiguous().data.view(-1).tolist()
        
        word_vecs = self.dropout(self.word_lut(inputs))
        packed_word_vecs = pack(word_vecs, lengths)
        rnn_output, hidden = self.lstm(packed_word_vecs)
        rnn_output = pad(rnn_output)[0]
        rnn_output = self.dropout(rnn_output)

        output_flat = self.linear_output(
                rnn_output.view(
                    rnn_output.size(0) * rnn_output.size(1), 
                    rnn_output.size(2)
                    )
                )
        
        return output_flat 



