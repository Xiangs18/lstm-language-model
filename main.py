import torch 
import data
from model import LanguageModel
from torch import nn
from torch.autograd import Variable
import time
import math
import os
import argparse
import shutil
def evaluate(
        model='model/LSTM_LM.pt',
        eval_data='data/penn/valid.txt',
        batch_size=64,
        data_reload=True,
        cuda=True
        ):
    
    if isinstance(model, str):
        model = torch.load(model)
    if cuda:
        model.cuda()

    if isinstance(eval_data, str):
        if os.path.exists(eval_data + '.preprocessed.pt') and data_reload:
            eval_dataset = torch.load(eval_data)
        else:
            eval_dataset = data.DataSet(eval_data, batch_size)
            torch.save(eval_dataset, eval_data + '.preprocessed.pt')
    else:
        eval_dataset = eval_data

    model.eval()
    criterion_weight = torch.ones(eval_dataset.num_vocb + 1)
    criterion_weight[0] = 0
    if opt.cuda:
        criterion = nn.CrossEntropyLoss(weight = criterion_weight, size_average=False).cuda()
    acc_loss = 0
    count = 0

    for batch_idx in range(eval_dataset.num_batch):
        batch_data, batch_lengths, target_words = eval_dataset[batch_idx]
        if cuda:
            batch_data = batch_data.cuda()
            batch_lengths = batch_lengths.cuda()
            target_words = target_words.cuda()
        #Forward
        output = model.forward(batch_data, batch_lengths)
        loss = criterion(output, target_words.view(-1))
        acc_loss += loss.data
        count += batch_lengths.data.sum()
    return acc_loss[0] / count 

def train(opt):
    
    if opt.data_reload and os.path.exists(opt.train_data + '.preprocessed.pt'):
        print('='*89)
        print('There is preprocessed training data, loading...')
        train_dataset = torch.load(opt.train_data + '.preprocessed.pt')
        print('Done')
        train_dataset.describe_dataset()
    else:
        print('='*89)
        print('Make training data')
        train_dataset = data.DataSet(opt.train_data, opt.batch_size, build_dict=True)
        torch.save(train_dataset, opt.train_data + '.preprocessed.pt')
        print('Save preprocessed training dataset at %s'%(opt.train_data + '.preprocessed.pt'))

    if opt.data_reload and os.path.exists(opt.val_data + '.preprocessed.pt'):
        print('='*89)
        print('There is preprocessed validation data, loading...')
        val_dataset = torch.load(opt.val_data + '.preprocessed.pt')
        print('Done')
        val_dataset.describe_dataset()
    else:
        print('='*89)
        print('Make validation data')
        val_dataset = data.DataSet(opt.val_data, opt.val_batch_size)
        torch.save(val_dataset, opt.val_data + '.preprocessed.pt')
        print('Save preprocessed validation dataset at %s'%(opt.val_data + '.preprocessed.pt'))

    val_dataset.change_dict(train_dataset.dictionary) 

    model = LanguageModel(
            train_dataset.num_vocb,
            dim_word = opt.dim_word,
            dim_rnn = opt.dim_rnn,
            num_layers = opt.num_layers,
            dropout_rate = opt.dropout_rate
            )
    #print('='*89)
    #print('Print language model')
    #print(model)
    if opt.cuda:
        model.cuda()
        print('Using GPU %d'%torch.cuda.current_device())
    else:
        print('Using CPU')

    model.train()
    
    criterion_weight = torch.ones(train_dataset.num_vocb + 1)
    criterion_weight[0] = 0
    if opt.cuda:
        criterion = nn.CrossEntropyLoss(weight = criterion_weight, size_average=False).cuda()
    
    acc_loss = 0
    start_time = time.time()
    
    best_model = {'val_loss' : 100, 'val_ppl' : math.exp(100), 'epoch' : 1}

    print('='*89)
    print(' ')
    print('Start training, will go through %d epoch'%opt.epoch)
    print(' ')
    for epoch_idx in range(opt.epoch):
        acc_loss = 0
        count = 0
        print('='*89)
        print('Start epoch %d, learning rate %f '%(epoch_idx + 1, opt.lr))
        epoch_start_time = start_time
        for batch_idx in range(train_dataset.num_batch):
            #Generate data
            batch_data, batch_lengths, target_words = train_dataset[batch_idx]
            if opt.cuda:
                batch_data = batch_data.cuda()
                batch_lengths = batch_lengths.cuda()
                target_words = target_words.cuda()
                #batch_mask = batch_mask.cuda()

            #Forward
            output_flat = model.forward(batch_data, batch_lengths)
            #Backward
            loss = criterion(output_flat, target_words.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
            for param in model.parameters():
                param.data.add_(-opt.lr * param.grad.data)

            acc_loss += loss.data
            count += batch_lengths.data.sum()

            if batch_idx % opt.display_freq == 0 and batch_idx > 0:
                average_loss = acc_loss[0] / count
                print('Epoch : %d, Batch : %d / %d, Loss : %f, Perplexity : %f, Time : %f'%(
                    epoch_idx + 1, 
                    batch_idx, train_dataset.num_batch, 
                    average_loss, 
                    math.exp(average_loss),
                    time.time() - start_time))
                acc_loss = 0
                count = 0
                start_time = time.time()

        print('Epoch %d finished, spend %d s'%(epoch_idx + 1,time.time() - epoch_start_time))
         
        val_loss = evaluate(
                model=model,
                eval_data=val_dataset,
                batch_size=opt.val_batch_size,
                data_reload=opt.data_reload
                )

        print('-'*89) 
        print('Validation Loss : %f'%val_loss)
        print('Validation Perplexity : %f'%math.exp(val_loss))

        model.val_loss = val_loss
        model.val_ppl = math.exp(val_loss)
        model.epoch_idx = epoch_idx + 1


        if (1 + epoch_idx) % opt.save_freq == 0:
            model_save = opt.model_name + '-e_' + str(epoch_idx + 1) +'-ppl_' + str(int(math.exp(val_loss))) + '.pt'
            torch.save(model, model_save)
            print('-'*89)
            print('Save model at %s'%(model_save))  

        if model.val_loss < best_model['val_loss']:
            print('-'*89)
            print('New best model on validation set')
            best_model['val_loss'] = model.val_loss
            best_model['val_ppl'] = model.val_ppl
            best_model['epoch_idx'] = model.epoch_idx
            best_model['name'] = model_save

    print('='*89)
    print('Finish training %d epochs!'%opt.epoch)
    print('-'*89)
    print('Best model:')
    print('Epoch : %d, Loss : %f, Perplexity : %f'%(best_model['epoch_idx'], best_model['val_loss'], best_model['val_ppl']))
    print('-'*89)
    print('Save best model at %s'%(opt.model_name + '.best.pt'))
    shutil.copy2(best_model['name'], opt.model_name + '.best.pt')
    print('='*89)

        
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
            help='Save model every several epoch')
    parser.add_argument('--cuda', action='store_true',
            help='Use cuda or not')
    parser.add_argument('--clip', type=float, default=5,
            help='Prevent gradient explode')
    
    
    opt = parser.parse_args()
    print('='*89)
    print('Configurations')
    for arg, value in vars(opt).iteritems():
        print(arg, value)
    
    train(opt)
