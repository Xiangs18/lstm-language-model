import torch 
import data
from model import LanguageModel
from torch import nn
import time
import math
import os
import argparse
def validate(
        model='model/LSTM_LM.pt',
        data='data/penn/valid.txt',
        batch_size=64,
        data_reload=True,
        cuda=True
        ):
    
    if isinstance(model, str):
        model = torch.load(model)
    
    if os.path.exists(data + '.preprocessed.pt') and data_reload:
        val_dataset = torch.load(data)
    else:
        val_dataset = data.DataSet(data)
        torch.save(val_dataset, data + '.preprocessed.pt')

    model.eval()

    for batch_idx in range(val_dataset.num_batch):
        batch_data, batch_lengths, target_words = train_dataset[epoch_idx]
        if cuda:
            batch_data = batch_data.cuda()
            batch_lengths = batch_lengths.cuda()
        #Forward
        output = model.forward(batch_data, batch_lengths)
        

def train(opt):
    
    if opt.data_reload and os.path.exists(opt.train_data + '.preprocessed.pt'):
        print('='*89)
        print('There is preprocessed training data, loading...')
        train_dataset = torch.load(opt.train_data + '.preprocessed.pt')
        print('Done')
        train_dataset.description()
    else:
        train_dataset = data.DataSet(opt.train_data, opt.batch_size)
        torch.save(train_dataset, opt.train_data + '.preprocessed.pt')
        print('Save preprocessed dataset at %s'%(opt.train_data + '.preprocessed.pt'))

    model = LanguageModel(
            train_dataset.num_vocb,
            dim_word = opt.dim_word,
            dim_rnn = opt.dim_rnn,
            num_layers = opt.num_layers,
            dropout_rate = opt.dropout_rate
            )
    if opt.cuda:
        model.cuda()

    model.train()

    criterion = nn.CrossEntropyLoss()
    acc_loss = 0
    start_time = time.time()

    print('='*89)
    print('Training, will go through %d epoch'%opt.epoch)
    print(' ')
    for epoch_idx in range(opt.epoch):
        acc_loss = 0
        print('='*89)
        print('Start epoch %d '%epoch_idx)
        epoch_start_time = start_time
        for batch_idx in range(train_dataset.num_batch):
            #Generate data
            batch_data, batch_lengths, target_words = train_dataset[epoch_idx]
            if opt.cuda:
                batch_data = batch_data.cuda()
                batch_lengths = batch_lengths.cuda()
                target_words = target_words.cuda()

            #Forward
            output = model.forward(batch_data, batch_lengths)
            #Backward
            model.zero_grad()
            loss = criterion(output, target_words)
            loss.backward()

            for param in model.parameters():
                param.data.add_(-opt.lr * param.grad.data)

            acc_loss += loss.data
            if batch_idx % opt.display_freq == 0 and batch_idx > 0:
                average_loss = acc_loss[0] / opt.display_freq
                print('Epoch : %d, Batch : %d / %d, Loss : %f, Perplexity : %f, Time : %f'%(
                    epoch_idx + 1, 
                    batch_idx, train_dataset.num_batch, 
                    average_loss, 
                    math.exp(average_loss),
                    time.time() - start_time))
                acc_loss = 0
                start_time = time.time()

        print('Epoch %d finished, spend %d s'%(epoch_idx + 1,time.time() - epoch_start_time))
        if (1 + epoch_idx) % opt.save_freq == 0:
            torch.save(model, opt.model_name + 'e' + str(epoch_idx + 1) + '.pt')
        '''
        val_loss = validate(
                model=model,
                data=opt.val_data,
                batch_size=opt.val_batch_size,
                data_reload=opt.data_reload
                )
        '''
        #print('Validation Loss : %f'%val_loss)
        #print('Validation Perplexity : %f'%math.exp(val_loss))

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LSTM language model')
    
    parser.add_argument('--train_data', type=str, default='./data/penn/train.txt',
            help='Training data path')
    parser.add_argument('--val_data', type=str, default='./data/penn/valid.txt')
    parser.add_argument('--model_name', type=str, default='LSTM_LM',
            help='Model name')
    parser.add_argument('--dim_word', type=int, default=256,
            help='Dimension of word embeddig vector')
    parser.add_argument('--dim_rnn', type=int, default=512,
            help='Dimension of LSTM')
    parser.add_argument('--batch_size',type=int, default=64,
            help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=64,
            help='Validation batch size')
    parser.add_argument('--epoch', type=int, default=10,
            help='Finish after several epochs')
    parser.add_argument('--model_reload', action='store_true',
            help='Relaod model')
    parser.add_argument('--data_reload', action='store_true',
            help='Reload preprocessed data')
    parser.add_argument('--optimizer', type=str, default='sgd',
            help='type of optimizer')
    parser.add_argument('--lr', type=float, default=0.1,
            help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=1,
            help='Number of LSTM layers')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
            help='Dropout rate')
    parser.add_argument('--display_freq', type=int, default=10,
            help='Display every several bathces')
    parser.add_argument('--save_freq', type=int, default=10,
            help='Save model every 