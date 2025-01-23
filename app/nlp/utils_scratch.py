import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
import datasets, torchtext
from torchtext.vocab import build_vocab_from_iterator
import math

def tokenize_data(example):
    return {'tokens': tokenizer(example[column_name])}

column_name = 'Joke'

seed=69
tokenizer = get_tokenizer('basic_english')
dataset_name = "Maximofn/short-jokes-dataset"
dataset = datasets.load_dataset(dataset_name)

train_test_split  = dataset['train'].train_test_split(test_size=0.3, seed=seed)
validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=seed)
# recombine into DatasetDict for easier usage
split_dataset = {
    "train": train_test_split["train"],
    "validation": validation_test_split["train"],
    "test": validation_test_split["test"],
}
split_dataset = datasets.DatasetDict(split_dataset)
train_dataset = split_dataset["train"]
tokenized_train_dataset = train_dataset.map(tokenize_data, remove_columns=[column_name])
vocab = build_vocab_from_iterator(tokenized_train_dataset['tokens'], min_freq=3)
vocab.insert_token('<unk>', 0)
vocab.insert_token('<eos>', 1)
vocab.set_default_index(vocab['<unk>'])

vocab_size = len(vocab)
emb_dim = 1024                # 400 in the paper
hid_dim = 1024                # 1150 in the paper
num_layers = 2                # 3 in the paper
dropout_rate = 0.65              
lr = 1e-3     
max_seq_len = 30

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,   
                self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        #src: [batch_size, seq len]
        embedding = self.dropout(self.embedding(src)) #harry potter is
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden)
        #ouput: [batch size, seq len, hid dim]
        #hidden: [num_layers * direction, seq len, hid_dim]
        output = self.dropout(output)
        prediction =self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden