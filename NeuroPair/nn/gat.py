import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Functions
class AttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size, n_behaviour, num_neurons):
        super(AttentionHead, self).__init__() 

        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.n_behaviour = n_behaviour
        self.num_neurons= num_neurons

        self.pred_layer = self.pred_layer_net() 
        self.neuron_behaviour_alpha = nn.Parameter(torch.rand(num_neurons))  
        self.mlp = self.build_combining_network()  

    def build_combining_network(self):
        """Define the combining neural network architecture."""
        layers = [ 
            nn.Linear(self.num_neurons, self.hidden_dim * 2),  
            nn.ReLU(), 
            # nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_neurons)
        ]
        return nn.Sequential(*layers)

    def pred_layer_net(self):
        """Define the prediction neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_behaviour), 
        )  
  
    def forward(self, x):   

        alpha = self.neuron_behaviour_alpha.softmax(-1)    
 
        out = []
        for neuron in range(x.shape[1]):
            ax = x[:, neuron, :]*alpha[neuron]
            out.append(ax.unsqueeze(1)) 

        out = torch.cat(out, dim=1)     
        return self.pred_layer(out).squeeze(-1), alpha.squeeze(0).repeat(x.shape[0], 1)     
          

class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features, out_features, n_heads):
        """
        Implementaion of a multi-head graph attention mechanism. 
        
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node
            n_heads (int): Number of attention heads to use for parallel attention processes
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, zh, adj_mat):
        """
        Propagates the input through the graph attention layer to compute the output features and attention scores.

        Args:
            zh (torch.tensor): Node features tensor,  
            adj_mat (torch.tensor): Adjacency matrix of the graph

        Returns:
            torch.tensor: Aggregated node features after applying attention 
            torch.tensor: Attention scores for each node pair
        """
        
        g_l = self.linear_l(zh).view(zh.shape[0], self.n_heads, self.n_hidden)
        g_r = self.linear_r(zh).view(zh.shape[0], self.n_heads, self.n_hidden)

        # Create repeated arrays
        g_l_repeat = g_l.repeat(zh.shape[0], 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(zh.shape[0], dim=0)

        # Sum features from transformations
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(zh.shape[0], zh.shape[0], self.n_heads, self.n_hidden)

        # Compute attention scores 
        e = self.attn(self.activation(g_sum)).squeeze(-1)
        att_score = e.masked_fill(adj_mat == 0, float('-inf'))

        # Apply softmax to normalize the attention scores
        att_score_sm = self.softmax(att_score)
        attn_res = torch.einsum('ijh,jhf->ihf', att_score_sm, g_r)

        return attn_res.mean(dim=1)