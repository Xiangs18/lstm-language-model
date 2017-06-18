import torch 
import data
from model import LanguageModel

def validate
def train(
        train_data='data/penn/train.txt',
        val_data='data/penn/validate.txt',
        batch_size=64,
        val_batch_size=64,
        model_name='LSTM_LM',
        reload_=True,
        optimizer='sgd',
        lr=0.1,
        epoch=20,
        preprocess='reload',
        cuda=True,
        dim_word=256,
        dim_rnn=512,
        num_layers=1,
        dropout_rate=0.3
        ):

    train_dataset = data.DataSet(train_dataset, batch_size)
    val_dataset = data.DataSet(val_data, val_batch_size)

    model = LanguageMode(
            train_dataset.num_vocb,
            dim_word = dim_word,
            dim_rnn = dim_rnn,
            num_layers = num_layers,
            dropout_rate = dropout_rate
            )
    for epoch_idx in range(epoch):
        batch_data, batch_lengths = train_dataset[epoch_idx]
        output = model.forward(batch_data;)


if __name__ == '__main__':
    
    train(arg)
