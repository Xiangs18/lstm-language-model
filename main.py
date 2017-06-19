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
import re
from evaluate import evaluate

def print_line(sym='=',width=89):
    print(sym*width)

def train(opt):

    # Read preprocessed data
    print_line()
    print('Loading training data ...')
    check_name = re.compile('.*\.prep\.train\.pt')
    assert os.path.exists(opt.train_data) or check_name.match(opt.train_data) is None
    train_dataset = torch.load(opt.train_data)
    train_dataset.set_batch_size(opt.batch_size)
    print('Done.')

    print_line()
    print('Loading validation data ...')
    check_name = re.compile('.*\.prep\.val\.pt')
    assert os.path.exists(opt.val_data) or check_name.match(opt.val_data) is None
    val_dataset = torch.load(opt.val_data)
    val_dataset.set_batch_size(opt.batch_size)
    print('Done.')

    # Build / load  Model
    if opt.model_reload is None:
        print_line()
        print('Build new model...')

        model = LanguageModel(
                train_dataset.num_vocb,
                dim_word = opt.dim_word,
                dim_rnn = opt.dim_rnn,
                num_layers = opt.num_layers,
                dropout_rate = opt.dropout_rate
                )
        model.dictionary = train_dataset.dictionary
        print('Done')
        train_dataset.describe_dataset()
        val_dataset.describe_dataset()
        
    else:
        print_line()
        print('Loading existing model...')
        model = torch.load(opt.model_reload)
        print('done')
        train_dataset.change_dict(model.dictionary)
        val_dataset.change_dict(model.dictionary)

    print_line()
    if opt.cuda:
        model.cuda()
        print('Using GPU %d'%torch.cuda.current_device())
    else:
        print('Using CPU')

    model.train()
    # Crterion, mask padding
    criterion_weight = torch.ones(train_dataset.num_vocb + 1)
    criterion_weight[0] = 0
    if opt.cuda:
        criterion = nn.CrossEntropyLoss(weight = criterion_weight, size_average=False).cuda()

    model_start_epoch = model.epoch_idx
    model_start_batch = model.batch_idx
    
    print_line()
    print(' ')
    if opt.model_reload is None:
        print('Start training new model, will go through %d epoch'%opt.epoch)
    else:
        print('Continue existing model, from epoch %d, batch %d to epoch %d'%(model_start_epoch, model_start_batch, opt.epoch))
    print(' ')

    lr = opt.lr
    best_model = {'val_loss' : 100, 'val_ppl' : math.exp(100), 'epoch_idx' : 1, 'batch_idx' : 1}

    if opt.save_freq == 0:
        opt.save_freq = train_dataset.num_batch - 1

    if(model_start_epoch > opt.epoch):
        print('This model has already trained more than %d epoch, add epoch parameter is you want to continue'%(opt.epoch + 1))
        return

    for epoch_idx in range(model_start_epoch, opt.epoch):
        # New epoch
        acc_loss = 0
        acc_count = 0
        start_time = time.time()
        train_dataset.shuffle_batch()

        print_line()
        print('Start epoch %d, learning rate %f '%(epoch_idx + 1, lr))
        print_line('-')
        epoch_start_time = start_time
        
        if epoch_idx == model_start_epoch and model_start_batch > 0:
            start_batch = model_start_batch
        else:
            start_batch = 0

        for batch_idx in range(start_batch, train_dataset.num_batch):
            # Generate batch data
            batch_data, batch_lengths, target_words = train_dataset[batch_idx]

            if opt.cuda:
                batch_data = batch_data.cuda()
                batch_lengths = batch_lengths.cuda()
                target_words = target_words.cuda()
            
            batch_data = Variable(batch_data)
            batch_lengths = Variable(batch_lengths)
            target_words = Variable(target_words)

            # Forward
            output_flat = model.forward(batch_data, batch_lengths)

            # Backward
            loss = criterion(output_flat, target_words.view(-1))
 
            loss.backward()

            # Prevent gradient explode
            torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)

            # Update parameters
            # Need to add other optimizers
            for param in model.parameters():
                param.data.add_(-opt.lr * param.grad.data)

            # Accumulate loss
            acc_loss += loss.data
            acc_count += batch_lengths.data.sum()

            # Display progress
            if batch_idx % opt.display_freq == 0:
                average_loss = acc_loss[0] / acc_count
                print('Epoch : %d, Batch : %d / %d, Loss : %f, Perplexity : %f, Time : %f'%(
                    epoch_idx + 1, 
                    batch_idx, train_dataset.num_batch, 
                    average_loss, 
                    math.exp(average_loss),
                    time.time() - start_time))

                acc_loss = 0
                acc_count = 0
                start_time = time.time()

            #Save and validate if it is neccesary
            if (1 + batch_idx) % opt.save_freq == 0:

                print_line('-') 
                print('Pause training for save and validate.')

                model.eval()
                val_loss = evaluate(
                    model=model,
                    eval_dataset=val_dataset,
                    cuda=opt.cuda
                    )
                model.train()

                print('Validation Loss : %f'%val_loss)
                print('Validation Perplexity : %f'%math.exp(val_loss))

                model.val_loss = val_loss
                model.val_ppl = math.exp(val_loss)
                model.epoch_idx = epoch_idx + 1
                model.batch_idx = batch_idx + 1
                
                model_save = opt.model_name + '-e_' + str(epoch_idx + 1) + '-b_' + str(batch_idx + 1) +'-ppl_' + str(int(math.exp(val_loss))) + '.pt'
                torch.save(model, model_save)
                
                if model.val_loss < best_model['val_loss']:
                    print_line('-')
                    print('New best model on validation set')
                    best_model['val_loss'] = model.val_loss
                    best_model['val_ppl'] = model.val_ppl
                    best_model['epoch_idx'] = model.epoch_idx
                    best_model['batch_idx'] = model.batch_idx
                    best_model['name'] = model_save
                    shutil.copy2(best_model['name'], opt.model_name + '.best.pt')


                print_line('-')
                print('Save model at %s'%(model_save))
                print_line('-')
                print('Continue Training...')
        print_line('-')
        print('Epoch %d finished, spend %d s'%(epoch_idx + 1,time.time() - epoch_start_time))

        
        lr *= opt.lr_decay

    print_line()
    print(' ')
    print('Finish training %d epochs!'%opt.epoch)
    print(' ')
    print_line()
    print('Best model:')
    print('Epoch : %d, Batch : %d ,Loss : %f, Perplexity : %f'%(
        best_model['epoch_idx'],
        best_model['batch_idx'], 
        best_model['val_loss'], 
        best_model['val_ppl']))
    print_line('-')

    print('Save best model at %s'%(opt.model_name + '.best.pt'))
    shutil.copy2(best_model['name'], opt.model_name + '.best.pt')
    print_line()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LSTM language model')
    
    parser.add_argument('--train_data', type=str, default='./data/penn/train.txt',
            help='Training data path')
    parser.add_argument('--val_data', type=str, default='./data/penn/valid.txt',
            help='Validation data path')
    parser.add_argument('--model_name', type=str, default='penn-lm',
            help='Model name')
    parser.add_argument('--model_reload', type=str, default=None,
            help='Relaod model')
    parser.add_argument('--dim_word', type=int, default=256,
            help='Dimension of word embeddig vector')
    parser.add_argument('--dim_rnn', type=int, default=512,
            help='Dimension of LSTM')
    parser.add_argument('--num_layers', type=int, default=1,
            help='Number of LSTM layers')
    parser.add_argument('--batch_size',type=int, default=64,
            help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=64,
            help='Validation batch size')
    parser.add_argument('--epoch', type=int, default=10,
            help='Finish after several epochs')
    parser.add_argument('--cuda', action='store_true',
            help='Use cuda or not')
    parser.add_argument('--optimizer', type=str, default='sgd',
            help='type of optimizer')
    parser.add_argument('--lr', type=float, default=0.1,
            help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0,
            help='Learning rate decay every epoch')
    parser.add_argument('--clip', type=float, default=5,
            help='Prevent gradient explode')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
            help='Dropout rate')
    parser.add_argument('--display_freq', type=int, default=10,
            help='Display every several bathces')
    parser.add_argument('--save_freq', type=int, default=10,
            help='Save model every several epoch')
    parser.add_argument('--random_seed', type=int, default=111,
            help='Random seed to reproduce result')
    
    
    opt = parser.parse_args()

    print_line()
    print('Configurations')
    for arg, value in vars(opt).iteritems():
        print(arg, value)
    
    train(opt)
