import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn.pack_padded_sequence as pack
import torch.nn.utils.rnn.pad_packed_sequence as pad
class LanguageModel(nn.Module):
    def __init__(self, voc_size, dim_word, dim_rnn, num_layers, dropout_rate = 0.2):
        super(LanguageModel, self).__init__()
        self.rnn_layers = num_layers
        self.dim_rnn = dim_rnn

        self.dropout = nn.Dropout(dropout_rate)
        self.word_lut = nn.Embedding(voc_size, dim_word)
        self.lstm = nn.LSTM(dim_word, dim_rnn, num_layers, dropout=dropout_rate)
        self.output = nn.Linear(dim_rnn, voc_size)

        self.init_weights()

    def init_weights(self, std = 0.1):
        self.word_lut.weight.data.normal_(mean=0, std=std)
        self.lstm.weight.data.normal_(mean=0, std=std)
        self.output.data.normal_(mean=0, std=std)

    def forward(self, inputs, lengths):
        inputs = pack(inputs, lengths)
        emb = self.dropout(self.word_lut(inputs))
        hidden = self.init_hidden(inputs.data.size(1))
        encode, hidden = self.lstm(emb, hidden)
        encode = pad(encode)[0]
        encode = self.dropout(encode)
        output = self.output(encode.view(encode.size(0) * encode.size(1), encode.size(2)))
        return output.view(encode.size(0), encode.size(1), encode.size(2))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.dim_rnn))

