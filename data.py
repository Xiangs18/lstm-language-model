import torch
import os
import random
import math
import time
from torch.autograd import Variable
from collections import OrderedDict
class DataSet:
    def __init__(self, datapath, batch_size):
        self.dictionary = {}
        self.frequency = {}
        self.sentence = []
        self.batch_size = batch_size
        self.datapath = datapath

        print('='*89)
        print('Loading data and preprocess data from %s ...'%datapath)
        start_time = time.time()
        # Build dictionary
        with open(datapath, 'r') as f:
            self.num_tokens = 0
            self.num_vocb = 0

            for line in f:
                tokens = line.split() + ['<eos>']
                self.num_tokens += len(tokens)
                for token in tokens:
                    if token not in self.frequency:
                        self.frequency[token] = 1 
                        self.num_vocb += 1
                    else:
                        self.frequency[token] += 1
            
            self.frequency['<unk>'] = 0
            self.frequency = OrderedDict(sorted(self.frequency.items(), key=lambda x : x[1], reverse=True))
             
            self.dictionary = OrderedDict(zip(self.frequency.keys(), range(self.num_vocb)))
        

        #Convert tokens to integers
        with open(datapath, 'r') as f:
            for line in f:
                tokens = line.split() + ['<eos>']
                sequence = [self.dictionary[token] for token in tokens]
                self.sentence.append(torch.LongTensor(sequence))
        
        print('Finishing loading and preprocessing data in %f s.'%(time.time()-start_time))
        
        print('Save dictionary at %s.dict'%datapath)

        with open(datapath + '.dict', 'w+') as f:
            for token, number in self.dictionary.iteritems():
                f.write('%s %d\n'%(token,number))


        self.num_batch = int(len(self.sentence) / self.batch_size)

        self.shuffle = range(self.__len__())
        random.shuffle(self.shuffle)

        self.describe_dataset()

    def describe_dataset(self):
        print('='*89)
        print('Data discription:')
        print('Data name : %s'%self.datapath)
        print('Number of batches : %d'%self.num_batch)
        print('Number of sentence : %d'%len(self.sentence))
        print('Number of tokens : %d'%self.num_tokens)
        print('Vocabulary size : %d'%self.num_vocb)

    def shuffle_batch(self):
        self.shuffle(self.shuffle)


    def get_batch(self, batch_idx):
        lengths = [self.sentence[x].size(0) for x in range(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))]
        max_len = max(lengths)
        total_len = sum(lengths)

        sorted_lengths = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)
        
        batch_data = torch.LongTensor(self.batch_size, max_len)
        batch_data.zero_()

        #target_words = torch.FloatTensor(total_len, self.num_vocb)
        #target_words.zero_()
        #target_offset = 0
        
        target_words = torch.LongTensor(self.batch_size, max_len)
        target_words.zero_()

        for i in range(self.batch_size):
            len_ = sorted_lengths[i][1] 
            idx_ = sorted_lengths[i][0]

            sequence_idx = idx_ + self.batch_size * batch_idx
            batch_data[i].narrow(0, 0, len_).copy_(self.sentence[sequence_idx])
            target_words[i].narrow(0, 1, len_ - 1 ).copy_(self.sentence[sequence_idx][1:])

            #one hot
            #target_words.narrow(1, i, 1).narrow(0, 1, len_ - 1).squeeze().scatter_(1, batch_data[i].narrow(0, 1, len_ - 1).view(-1, 1), 1)
    

            
        return Variable(batch_data.t()), \
                Variable(torch.LongTensor([x[1] for x in sorted_lengths])), \
                Variable(target_words.t().contiguous().view(-1))


    def __getitem__(self, index):
        return self.get_batch(index) 

    def __len__(self):
        return self.num_batch

#Test
if __name__ == '__main__':
    test_data_path = 'data/penn/test.txt'
    test_dataset = DataSet(test_data_path, batch_size = 64)
    batch_data,_,target = test_dataset[0]
    print(target.data.size())
    for i in range(len(test_dataset)):
        batch_data, lengths, target = test_dataset[i]
        #if(lengths.data.min() <=0):
        print(lengths.contiguous().data.view(-1).tolist())
        
