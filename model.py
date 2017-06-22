import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import math
class LanguageModel(nn.Module):

    def __init__(self, voc_size, dim_word, dim_rnn, num_layers, dropout_rate = 0.2):
        
        super(LanguageModel, self).__init__()
        # Hyperparamters
        self.rnn_layers = num_layers
        self.dim_rnn = dim_rnn
        self.num_layers = num_layers
        
        # Layers
        self.dropout = nn.Dropout(dropout_rate)
        self.word_lut = nn.Embedding(voc_size + 1, dim_word, padding_idx=0)
        self.lstm = nn.LSTM(dim_word, dim_rnn, num_layers, dropout=dropout_rate)
        self.output = nn.Linear(dim_rnn, voc_size + 1)
        self.logprob = nn.LogSoftmax()
        
        # Model train status
        self.train_info = {}
        self.train_info['val loss'] = 100
        self.train_info['train loss'] = 100
        self.train_info = ['epoch idx'] = 0
        self.train_info = ['batch idx'] = 0
        self.train_info['val ppl'] = math.exp(self.val_loss)
        
        # Dictionary for token to index
        self.dictionary = None


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



