# lstm-language-model
Implementation of LSTM language model using PyTorch

There is a example for Penn Treebank dataset

## preprocess.py
Preprocess the data before training and evaluate and save the data into a PyTorch data structure. \\
Nececcary before training\\

Parameters | Description
------------ | -------------
--train_data | Training data, default='data/penn/train.txt'
--val_data | Validation data, default='data/penn/valid.txt'
--dict_size | Reduce dictionary if overthis size, default=50000
--display_freq | Display progress every this number of sentences, 0 for no diplay, default=100000
--max_len | Maximum length of sentence, default=100,
--trunc_len |Truncate the sentence that longer than maximum length, default=100

## train.py
Training language model

Parameters | Description
-- | --
--train_data | Training data path, default= './data/penn/train.txt.prep.train.pt' 
--val_data | Validate data path, default= './data/penn/valid.txt.prep.val.pt' 
--model_name | Model name, default= 'model/exp8-lstm-lm' 
--reload_name | Training from a existing model
--dim_word | Dimension of word embedding, default= 200 
--dim_rnn  | Dimension of LSTM (or RNN), default= 200
--num_layers | Number of layer os LSTM(or RNN), default= 2 
--batch_size | Training batch size, default= 64 
--val_batch_size | Validate batch size, default= 64 
--epoch | Maximum epoch, default= 10 
--optimizer | Optimizer type [SGD, Adam, Adadelta], default= SGD 
--lr | Learning rate, default= 1 
--lr_decay | Learning rate decay for every epoch, default= 0.9 
--clip | Clip gradient to prevent gradient explode, default= 5 
--dropout_rate | Dropout rate, default= 0.3 
--display_freq | Display progress every several batches, default= 100 
--save_freq | Save the model every several batches, default= 0
--cuda | Set if use gpu

## evaluate.py
Evaluate performance of trained model

Parameters | Description
-- | --
--eval_data| Evaluate data path, default='./data/penn/test.txt' 
--model| Model, default='model/penn-lm.best.pt' 
--batch_size| Batch size, defalut=64
--cuda| Set if use GPU



