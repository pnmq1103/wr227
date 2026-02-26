import torch
import torch.nn as nn
import math
class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__() 
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):  
        return self.embedding(x)*math.sqrt(self.d_model)
class PositionalEncoding(nn.Module): 
    def __init__(self, d_model :int , seq_len: int, drop_out: float)->None: 
         super().__init__()
         self.d_model =  d_model
         self.seq_len = seq_len
         self.drop_out = nn.Dropout(drop_out)
         
         pe = torch.zeros(seq_len,d_model)

         position = torch.arange(0,seq_len, dtype= torch.float).unsqueeze(1)
         div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) 
         pe[:,0::2] =  torch.sin(position*div_term)
         pe[:,1::2] =  torch.cos(position*div_term)
         pe =  pe.squeeze(0)
def main(): 
    pa = PositionalEncoding(512,10,0.1)
    print(pa.pe.shape)
if __name__ == "__main__":
    main()

