import torch
import os
import random
import math
import time
from torch.autograd import Variable
from collections import OrderedDict
class DataSet:
    def __init__(self, datapath, batch_size=1, build_dict=False, display_freq=0):
        self.dictionary = {}
        self.frequency = {}
        self.sentence = []
        self.batch_size = batch_size
        self.datapath = datapath
        self.num_batch = 0
        self.num_tokens = 0
        self.num_vocb = 0
        self.shuffle = True
        self.display_freq = display_freq
        self.max_dict = 50000
        print('='*89)
        print('Loading data from %s ...'%datapath)
        
        if build_dict:
            self.build_dict()


    def describe_dataset(self):
        print('='*89)
        print('Data discription:')
        print('Data name : %s'%self.datapath)
        print('Number of sentence : %d'%len(self.sentence))
        print('Number of tokens : %d'%self.num_tokens)
        print('Vocabulary size : %d'%self.num_vocb)
        print('Number of batches : %d'%self.num_batch)
        print('Batch size : %d'%self.batch_size)


    def build_dict(self, save_as_text=True):
        print('Building dictionary...')
        with open(self.datapath, 'r') as f:
            self.num_tokens = 0
            self.num_vocb = 0
            
            for count, line in enumerate(f):
                if self.display_freq > 0 and count % self.display_freq == 0:
                    print('%d sentence processed'%(count))
                tokens = line.split() + ['<eos>']
                for token in tokens:
                    if token not in self.frequency:
                        self.frequency[token] = 1 
                        self.num_vocb += 1
                    else:
                        self.frequency[token] += 1
            
            self.frequency['<unk>'] = 1 + max(self.frequency.values())
            self.frequency = OrderedDict(sorted(self.frequency.items(), key=lambda x : x[1], reverse=True))
            
            if self.num_vocb > self.max_dict:
                self.num_vocb = self.max_dict
                self.frequency = self.frequency[:self.num_vocb]
            
            self.dictionary = OrderedDict(zip(self.frequency.keys(), range(1, self.num_vocb + 1)))
        
        print('Done.')
        
        print('Save dictionary at %s.dict'%self.datapath)

        with open(self.datapath + '.dict', 'w+') as f:
            for token, number in self.dictionary.iteritems():
                f.write('%s %d\n'%(token,number))

        self.index_token()
        

    def change_dict(self, dictionary):
        self.dictionary = dictionary
        self.num_vocb = len(dictionary)
        self.index_token()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batch = int(len(self.sentence) / self.batch_size)
        self.index = range(self.num_batch)
        #self.describe_dataset()

    def index_token(self):
        #Convert tokens to integers
        print('Index tokens ...')
        with open(self.datapath, 'r') as f:
            for count, line in enumerate(f):
                if self.display_freq > 0 and count % self.display_freq == 0:
                    print('%d  sentence processed'%(count))
                tokens = line.split() + ['<eos>']
                self.num_tokens += len(tokens)
                sequence = [self.dictionary[token] if token in self.dictionary else self.dictionary['<unk>'] for token in tokens]
                self.sentence.append(torch.LongTensor(sequence))
                
        self.num_batch = int(len(self.sentence) / self.batch_size)
        self.index = range(self.num_batch)
        self.describe_dataset()
        print('Done.')


    def get_batch(self, batch_idx):
        lengths = [self.sentence[x].size(0) for x in range(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))]
        max_len = max(lengths)
        total_len = sum(lengths)

        sorted_lengths = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)

        batch_data = torch.LongTensor(max_len, self.batch_size)
        batch_data.zero_()

        target_words = torch.LongTensor(max_len, self.batch_size)
        target_words.zero_()

        for i in range(self.batch_size):
            len_ = sorted_lengths[i][1] 
            idx_ = sorted_lengths[i][0]

            sequence_idx = idx_ + self.batch_size * batch_idx
            batch_data[: len_, i].copy_(self.sentence[sequence_idx])
            target_words[ : len_ - 1, i].copy_(self.sentence[sequence_idx][1 : len_])

        batch_lengths = torch.LongTensor([x[1] for x in sorted_lengths])

        return Variable(batch_data), Variable(batch_lengths), Variable(target_words)


    def shuffle_batch(self):
        assert self.shuffle, 'Enable shuffle first!'
        
        random.shuffle(self.index)


    def enable_shuffle(self, status):
        self.shuffle = status


    def __getitem__(self, index):
        if not self.shuffle:
            return self.get_batch(index) 
        else:
            return self.get_batch(self.index[index])

    def __len__(self):
        return self.num_batch

#Test
if __name__ == '__main__':
    test_data_path = 'data/penn/test.txt'
    test_dataset = DataSet(test_data_path, batch_size = 64)
    batch_data,_,target_oh = test_dataset[0]
    print(batch_data[:, 0])
    print(target_oh.data[:, 0])
    for i in range(len(test_dataset)):
        batch_data, lengths, target = test_dataset[i]
        #if(lengths.data.min() <=0):
        #print(batch_data.data)
        
