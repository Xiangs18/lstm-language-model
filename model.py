import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import math
class LanguageModel(nn.Module):

    def __init__(self, voc_size, dim_word, dim_rnn, num_layers, dropout_rate = 0.2):
        super(LanguageModel, self).__init__()
        self.rnn_layers = num_layers
        self.dim_rnn = dim_rnn
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout_rate)
        self.word_lut = nn.Embedding(voc_size + 1, dim_word, padding_idx=0)
        self.lstm = nn.LSTM(dim_word, dim_rnn, num_layers, dropout=dropout_rate)
        self.output = nn.Linear(dim_rnn, voc_size + 1)
        self.logprob = nn.LogSoftmax()

        self.val_loss = 100
        self.epoch_idx = 0
        self.val_ppl = math.exp(self.val_loss)


    def init_weights(self, std = 0.1):
        #self.word_lut.weight.data.normal_(mean=0, std=std)
        self.output.weight.data.normal_(mean=0, std=std)


    def forward(self, inputs, lengths):
        lengths = lengths.contiguous().data.view(-1).tolist()
        
        #hidden = self.init_hidden(inputs.data.size(1))
        hidden = None
        emb = self.dropout(self.word_lut(inputs))
        emb = pack(emb, lengths)
        encode, hidden = self.lstm(emb)
        encode = pad(encode)[0]
        encode = self.dropout(encode)

        output_flat = self.output(encode.view(encode.size(0) * encode.size(1), encode.size(2)))
        
        #Flatten the output
        return output_flat 


    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.dim_rnn))

