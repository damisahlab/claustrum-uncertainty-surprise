import torch
import torch.nn as nn 
from einops import rearrange  
import torch.nn.functional as F 
import numpy as np
 
from nn.gat import GraphAttentionV2Layer
from nn.lstm import RnnLSTM 
from nn.gat import AttentionHead

class NeuroPair(nn.Module):
    def __init__(
        self, 
        args, 
    ):
        """
        A single step (cell) of a Graph Recurrent Neural Network using
        GraphAttentionV2Layer + LSTM.

        Args:
            in_features (int): Number of input features per node.
            hidden_features (int): Number of hidden features per node.
            n_heads (int): Number of attention heads in the GAT layer.
        """
        super(NeuroPair, self).__init__()

        self.adj_factor = 1e-2 
        self.transform_type = 'none'
         
        self.device = args.device
        self.rnn_num_layers = args.rnn_num_layers 
        self.input_size = args.input_size
        self.hidden_size  = args.hidden_size
        self.dropout = args.dropout
        self.gat_n_heads = args.gat_n_heads 
        self.loss_type = 'mse' 

        # GAT layer for transforming input features
        self.gat1 = GraphAttentionV2Layer(
            in_features= self.input_size,
            out_features=self.hidden_size,
            n_heads=self.gat_n_heads
        ).to(self.device) 
          
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_num_layers,
            dropout=self.dropout if self.rnn_num_layers > 1 else 0,   
            batch_first=True
        ).to(self.device) 

        # GAT layer for transforming input features
        self.gat2 = GraphAttentionV2Layer(
            in_features= self.hidden_size,
            out_features=self.hidden_size,
            n_heads=self.gat_n_heads
        ).to(self.device) 

        self.attention_head = AttentionHead(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            n_behaviour=args.n_behaviour, 
            num_neurons=args.num_neurons
        ).to(self.device)  
       

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
        # import pdb;pdb.set_trace()
        # i) graph attention network      
        z_gat1 = [] 
        for i in range(x.shape[0]): 
            z_gat1.append(self.gat1(x[i,:,:], adj.unsqueeze(-1)).unsqueeze(0))  
        z_gat1 = torch.cat(z_gat1)  

        # ii) Recurrent update (GRUCell)  
        z_gat1 = rearrange(z_gat1, 't n d -> n t d')                                              # [324, 21, 40]
        h_0 = torch.zeros(self.rnn_num_layers, z_gat1.size(0), self.hidden_size).to(self.device)  # [2, 324, 128]
        c_0 = torch.zeros(self.rnn_num_layers, z_gat1.size(0), self.hidden_size).to(self.device)  # [2, 324, 128]
         
        z_rnn, (hidden, cell) = self.rnn(z_gat1, (h_0, c_0))                                      # [324, 21, 128]
        z_rnn = rearrange(z_rnn, 'n t d -> t n d')                                                # [324, 21, 40]

         # iii) graph attention network    
        z_gat2 = [] 
        for i in range(z_rnn.shape[0]): 
            z_gat2.append(self.gat2(z_rnn[i,:,:], adj.unsqueeze(-1)).unsqueeze(0))  
        z_gat2 = torch.cat(z_gat2)  
 
        out, neuron_behaviour_alpha = self.attention_head(z_gat2)  

        return  out, neuron_behaviour_alpha, z_gat2

    def loss(self, y, y_pred, reduction='sum'):  
        y = self.transform_targets(y.squeeze(-1))    
        if self.loss_type == 'mse': 
            y = y.flatten()
            y_pred = y_pred.flatten()
            loss = F.mse_loss(y_pred, y)
        elif self.loss_type == 'cosine': 
            y_pred = rearrange(y_pred, 'n d -> d n')  
            y = rearrange(y, 'n d -> d n')   
            cos_sim = F.cosine_similarity(y, y_pred)  
            loss = (1 - cos_sim)   
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean() 
        return loss

    def transform_targets(self, targets): 
        if self.transform_type == "log":
            targets = torch.log(targets + self.adj_factor) - self.adj_factor
        else: 
            targets = targets
        return targets

    def untransform_targets(self, preds): 
        if self.transform_type == "log":
            preds = torch.exp(preds + self.adj_factor) - self.adj_factor
        else: 
            preds = preds 
        return preds