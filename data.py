import torch
import os
import random
import math
import time
from torch.autograd import Variable
from collections import OrderedDict
import const
class DataSet:
    def __init__(self, datapath, batch_size=1, build_dict=False, display_freq=0, max_len=100, trunc_len=100):
        
        self.dictionary = {}
        self.frequency = {}
        self.sentence = []
        
        self.batch_size = batch_size
        self.datapath = datapath
        self.num_batch = 0
        self.num_tokens = 0
        self.num_vocb = 0
        self.shuffle_level = 2
        self.display_freq = display_freq
        self.max_dict = 50000
        self.max_len = max_len
        
        assert trunc_len <= max_len, 'trunc length should be smaller than maximum lenth'
        self.trunc_len = trunc_len
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

                tokens = [const.BOS_WORD] + line.split() + [const.EOS_WORD]
                
                for token in tokens:
                    if token not in self.frequency:
                        self.frequency[token] = 1 
                        self.num_vocb += 1
                    else:
                        self.frequency[token] += 1

            max_freq = max(self.frequency.values()) 
            self.frequency[const.UNK_WORD] = 4 - const.UNK + max_freq
            self.frequency[const.BOS_WORD] = 4 - const.BOS + max_freq
            self.frequency[const.EOS_WORD] = 4 - const.EOS + max_freq 
            self.frequency[const.PAD_WORD] = 4 - const.PAD + max_freq

            self.frequency = OrderedDict(sorted(self.frequency.items(), key=lambda x : x[1], reverse=True))
            
            if self.num_vocb > self.max_dict:
                self.num_vocb = self.max_dict
                self.frequency = self.frequency[:self.num_vocb]
            
            self.dictionary = OrderedDict(zip(self.frequency.keys(), range(1, self.num_vocb + 1)))
        
        print('Done.')
        
        print('Save dictionary at %s.dict'%self.datapath)

        with open(self.datapath + '.dict', 'w+') as f:
            for token, number in self.dictionary.items():
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
        self.sentence = []
        zero_sentence = 0
        long_sentence = 0
        with open(self.datapath, 'r') as f:
            for count, line in enumerate(f):
                
                if self.display_freq > 0 and count % self.display_freq == 0:
                    print('%d  sentence processed'%(count))
                
                tokens = line.split()
                
                if len(tokens) == 0:
                    zero_sentence += 1
                else:
                    if len(tokens) > self.max_len:
                        long_sentence += 1
                        if self.trunc_len > 0:
                            tokens = tokens[:self.trunc_len]
                        else:
                            continue
                    
                    self.num_tokens += len(tokens) - 2
                    tokens = [const.BOS_WORD] + tokens + [const.EOS_WORD]
                    sequence = [self.dictionary[token] if token in self.dictionary else self.dictionary[const.UNK_WORD] for token in tokens]
                    self.sentence.append(sequence)
    
        self.num_batch = int(len(self.sentence) / self.batch_size)
        self.index = range(self.num_batch)
        print('%d sentences were processed, %d longer than maximum length,%d were ignored because zero length'%(len(self.sentence), long_sentence, zero_sentence))
        self.describe_dataset()
        print('Done.')


    def get_batch(self, batch_idx):
        lengths = [len(self.sentence[x]) for x in range(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))]
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
            
            batch_data[: len_ - 1, i].copy_(torch.LongTensor(self.sentence[sequence_idx][: len_ - 1]))
            target_words[ : len_ - 1, i].copy_(torch.LongTensor(self.sentence[sequence_idx][1 : len_]))

        batch_lengths = torch.LongTensor([x[1] for x in sorted_lengths])

        return batch_data, batch_lengths, target_words


    def shuffle(self):
        print(self.shuffle_level)
        assert self.shuffle_level > 0, 'Enable shuffle first!'
        if self.shuffle_level == 1:
            random.shuffle(self.index)
        if self.shuffle_level == 2:
            random.shuffle(self.sentence)
    

    def change_shuffle_level(self, level):
        self.shuffle_level = level


    def __getitem__(self, index):
        if self.shuffle == 1:
            return self.get_batch(self.index[index]) 
        else:
            return self.get_batch(index)

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
        
