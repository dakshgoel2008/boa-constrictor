import torch
import torch.nn as nn

class SimpleGRUBackbone(nn.Module):
    def __init__(self, d_model=256, num_layers=4, vocab_size=256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.rnn = nn.GRU(input_size=d_model, 
                          hidden_size=d_model,
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  
        
        out, _ = self.rnn(embedded)   
        
        logits = self.fc_out(out)     
        return logits