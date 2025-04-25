 
import torch
import torch.nn as nn 
from einops import rearrange  
import torch.nn.functional as F 
import numpy as np
 
from nn.gat import GraphAttentionV2Layer
from nn.lstm import RnnLSTM 
from nn.gat import AttentionHead


class RecurrentGAT(nn.Module):
    def __init__(
        self, 
        args,
        # rnn_num_layers=2, 
        # input_size=40, 
        # hidden_size=128, 
        # dropout=0.2, 
        # gat_n_heads=4,
        # num_neurons=33, 
        # n_behaviour=21, 
        # device='cuda'
    ):
        """
        A single step (cell) of a Graph Recurrent Neural Network using
        GraphAttentionV2Layer + LSTM.

        Args:
            in_features (int): Number of input features per node.
            hidden_features (int): Number of hidden features per node.
            n_heads (int): Number of attention heads in the GAT layer.
        """
        super(RecurrentGAT, self).__init__()

        # Recurrent update over hidden state 
        self.device = args.device
        self.rnn_num_layers = args.rnn_num_layers 
        self.input_size = args.input_size
        self.hidden_size  = args.hidden_size
        self.dropout = args.dropout
        self.gat_n_heads = args.gat_n_heads 
 
        self.h0 = nn.Parameter(torch.zeros(self.rnn_num_layers, 1, self.input_size))
        self.c0 = nn.Parameter(torch.zeros(self.rnn_num_layers, 1, self.input_size))
          
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_num_layers,
            dropout=self.dropout if self.rnn_num_layers > 1 else 0,   
            batch_first=True
        ).to(self.device) 

        # GAT layer for transforming input features
        self.gat = GraphAttentionV2Layer(
            in_features= self.input_size,
            out_features=self.hidden_size,
            n_heads=self.gat_n_heads
        ).to(self.device) 

        self.attention_head = AttentionHead(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            output_size=args.n_behaviour, 
            num_neurons=args.num_neurons
        ).to(self.device) 


        self.h0 = nn.Parameter(torch.zeros(self.input_size, 1, self.input_size))
        self.c0 = nn.Parameter(torch.zeros(self.input_size, 1, self.input_size))
       

    def forward(self, x, adj):
        """
        Forward pass of the GraphRNN cell.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_features].
            adj (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes].
            h_prev (torch.Tensor): Previous hidden state of shape [num_nodes, hidden_features].

        Returns:
            h_new (torch.Tensor): Updated hidden state of shape [num_nodes, hidden_features].
        """
        # 2) Recurrent update (GRUCell) 
        x = rearrange(x, 't n d -> n t d')  

        h_0 = torch.zeros(self.rnn_num_layers, x.size(0), self.hidden_size).to(self.device)  # [2, 33, 40]
        c_0 = torch.zeros(self.rnn_num_layers, x.size(0), self.hidden_size).to(self.device)  # [2, 33, 40]
         
        z_rnn, (hidden, cell) = self.rnn(x, (h_0, c_0)) 
        z_rnn = rearrange(x, 'n t d -> t n d')
 
        z_gat = [] 
        for i in range(z_rnn.shape[0]): 
            z_gat.append(self.gat(z_rnn[i,:,:], adj.unsqueeze(-1)).unsqueeze(0))  
        z_gat = torch.cat(z_gat)    
        out, neuron_behaviour_alpha = self.attention_head(z_gat)  
        
        return  out, neuron_behaviour_alpha, z_gat

    def loss(self, y, y_pred, reduction='mean'):
        y = y.squeeze(-1) 
        y_pred = rearrange(y_pred, 't n d -> t (n d)') 
        # import pdb;pdb.set_trace()
        y = rearrange(y, 't n d -> t (n d)')  
        # 
        loss = F.mse_loss(y_pred, y)
        # cos_sim = F.cosine_similarity(y, y_pred)  
        # loss = (1 - cos_sim)   
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean() 
        return loss