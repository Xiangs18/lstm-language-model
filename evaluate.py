import torch
import argparse
import data
import os
import math
import torch.nn as nn

def evaluate(
        model=None,
        eval_dataset=None,
        cuda=True
        ):
     
    #assert model is None or eval_dataset is None
    
    criterion_weight = torch.ones(eval_dataset.num_vocb + 1)
    criterion_weight[0] = 0
    if cuda:
        criterion = nn.CrossEntropyLoss(weight = criterion_weight, size_average=False).cuda()
    
    acc_loss = 0
    acc_count = 0

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
        acc_count += batch_lengths.data.sum()
    return acc_loss[0] / acc_count 


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluate a trained model')

    parser.add_argument('--eval_data', type=str, default='./data/penn/test.txt',
            help='Path to evaluation data')

    parser.add_argument('--model', type=str, default='./model/penn-lm.best.pt',
            help='Path to trained model')

    parser.add_argument('--cuda', action='store_true',
            help='Use GPU')

    parser.add_argument('--batch_size', type=int, default=64,
            help='Evaluate batch size if use gpu')

    opt = parser.parse_args()

    #assert not os.path.exists(opt.model) or not os.path.exists(opt.eval_data)

    model = torch.load(opt.model)
    model.eval()
    
    eval_dataset = data.DataSet(opt.eval_data)
    eval_dataset.change_dict(model.dictionary)
   
    if opt.cuda:
        model.cuda()
        eval_dataset.set_batch_size(opt.batch_size)
    print('Start Evaluation ...')
    loss = evaluate(model, eval_dataset, opt.cuda)
    print('Evaluation Result')
    print('Loss : %f, Perplexity : %f'%(loss, math.exp(loss)))
    
