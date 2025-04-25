import torch
from torch import nn 
 

class autoregressive(nn.Module):
    def __init__(self, hparams):
        super(autoregressive, self).__init__()
        self.hparams = hparams
           
        # define the model  
        if hparams.autoregressive_model == 'LSTM':
            # from nn.lstm import RnnLSTM
            self.model = autoregressive(
                num_layers=hparams.num_layers, 
                seq_len=hparams.seq_len, 
                embedding_dim=hparams.embedding_dim, 
                n_features=hparams.n_features, 
                dropout=hparams.dropout
            ).to(hparams.device) 
        else:
            raise ValueError("Unsupported autoregressive model: {}".format(autoregressive_model))
 
        # define loss function 
        if hparams.loss_type == 'MSE':
            self.criterion = nn.MSELoss()  
        else:
            raise ValueError("Unsupported loss {}".format(loss_type))
        
        # define optimizer 
        if hparams.optim_type == 'ADAM':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=hparams.learn_rate, 
                weight_decay=hparams.weight_decay
            )
        else:
            raise ValueError("Unsupported optimizer {}".format(autoregressive_model))


    def forward(self, inputs):
        h_0 = torch.zeros(self.hparams.num_layers, inputs.size(0), self.hparams.embedding_dim).to(self.hparams.device)
        c_0 = torch.zeros(self.hparams.num_layers, inputs.size(0), self.hparams.embedding_dim).to(self.hparams.device)  
        output, _, _ = self.model(inputs, h_0, c_0)
        return output

    def loss(self, y, y_hat):
        return self.criterion(y_hat, y) 
 


 
