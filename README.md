# lstm-language-model
Implementation of LSTM language model using PyTorch
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
--train_data | default= './data/penn/train.txt.prep.train.pt' 
--val_data | default= './data/penn/valid.txt.prep.val.pt' 
--model_name | default= 'model/exp8-lstm-lm' 
--dim_word | default= 200 
--dim_rnn  | default= 200
--num_layers | default= 2 
--batch_size | default= 64 
--val_batch_size | default= 64 
--epoch | default= 10 
--optimizer | default= SGD 
--lr | default= 1 
--lr_decay | default= 0.9 
--dropout_rate | default= 0.3 
--display_freq | default= 100 
--save_freq | default= 0
--cuda | default=
--clip | default= 5 
## evaluate.py
Evaluate performance of trained model

## model
RNN model : LSTM, RNN, GRU
(TODO) n-gram model

An LSTM language model is in example
