import torch
import argparse
import data
def preprocess(opt):
    print('Begin preprocessing')

    train_dataset = data.DataSet(opt.train_data, display_freq=opt.display_freq)
    train_dataset.max_dict = opt.dict_size
    train_dataset.build_dict()
    print('Save training data')
    torch.save(train_dataset, opt.train_data + '.prep.train.pt')
    
    val_dataset = data.DataSet(opt.val_data, display_freq=opt.display_freq)
    val_dataset.change_dict(train_dataset.dictionary)
    print('Save validation data')
    torch.save(val_dataset, opt.val_data + '.prep.val.pt')
    
    print('Preprocessing done')
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Preprocessing')
    
    parser.add_argument('--train_data', type=str, default='data/penn/train.txt',
            help='Training data path')
    parser.add_argument('--val_data', type=str, default='data/penn/valid.txt',
            help='Validation data path')
    parser.add_argument('--dict_size', type=int, default=50000,
            help='Reduce dictionary if overthis size')
    parser.add_argument('--display_freq', type=int, default=100000,
            help='Display progress every this number of sentences, 0 for no diplay')
    parser.add_argument('--max_len', type=int, default=100,
            help='Maximum length od=f sentence')
    parser.add_argument('--trunc_len',type=int, default=100,
            help='Truncate the sentence that longer than maximum length')

    opt = parser.parse_args() 
    
    preprocess(opt)
