# lstm-language-model
Implementation of LSTM language model using PyTorch
## preprocess.py
Preprocess the data before training and evaluate

Parameters | Description
------------ | -------------
--train_data | Training data, default='data/penn/train.txt'
--val_data | Validation data pathtype=str, default='data/penn/valid.txt'
--dict_size | Reduce dictionary if overthis size, default=50000
--display_freq | Display progress every this number of sentences, 0 for no diplay, default=100000
--max_len | Maximum length of sentence, default=100,
--trunc_len |Truncate the sentence that longer than maximum length, default=100

## train.py
Training language model
## evaluate.py
Evaluate performance of trained model

## model
RNN model : LSTM, RNN, GRU
(TODO) n-gram model

An LSTM language model is in example
Result after %% epoch :  
