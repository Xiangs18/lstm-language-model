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


        print('Loading data and built dictionaryfrom %s ...'%datapath)
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
        
        print('Data discription:')
        print('Data name : %s'%datapath)
        print('Number of sentence : %d'%len(self.sentence))
        print('Number of tokens : %d'%self.num_tokens)
        print('Vocabulary size : %d'%self.num_vocb)
        print('Finishing loading data in %f s.'%(time.time()-start_time))
        
        print('Save dictionary at %s.dict'%datapath)

        with open(datapath + '.dict', 'w+') as f:
            for token, number in self.dictionary.iteritems():
                f.write('%s %d\n'%(token,number))

        
        self.num_batch = int(len(self.sentence) / self.batch_size)

        self.shuffle = range(self.__len__())
        random.shuffle(self.shuffle)


    def shuffle_batch(self):
        self.shuffle(self.shuffle)


    def get_batch(self, batch_idx):
        lengths = [self.sentence[x].size(0) for x in range(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))]
        max_len = max(lengths)
        
        batch_data = torch.zeros(self.batch_size, max_len)
        for i in range(self.batch_size):
            sequence_idx = i + self.batch_size * batch_idx
            batch_data[i].narrow(0, 0, lengths[i]).copy_(self.sentence[sequence_idx])

        return Variable(batch_data.t()), lengths


    def __getitem__(self, index):
        return self.get_batch(index) 

    def __len__(self):
        return self.num_batch

#Test
if __name__ == '__main__':
    test_data_path = 'data/penn/test.txt'
    test_dataset = DataSet(test_data_path, batch_size = 64)
    batch_data,_ = test_dataset[0]
    #print(batch_data.data)
    for i in range(len(test_dataset)):
        batch_data, lengths = test_dataset[i]
        #print(batch_data.size())
        
