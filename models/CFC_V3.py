import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv

class SelfAttention(nn.Module):
    """

    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        #print("size+",size)
        x = x.view(*size[:2], -1)
        #print("size-",x.size())
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
    
class CFC_V3_Model(nn.Module):
    def __init__(
        self,
        input_shape ,
        number_class , 
        filter_num = 16,
        filter_size = 5,
        nb_conv_layers = 4,
        dropout = 0.2,
        hidden_dim = 16,
        activation = "ReLU",
        sa_div= 1,
    ):
        super(CFC_V3_Model, self).__init__()
        
        # PART 1 , Channel wise Feature Extraction
        
        layers_conv = []
        for i in range(nb_conv_layers):
        
            if i == 0:
                in_channel = input_shape[1]
            else:
                in_channel = filter_num
    
            layers_conv.append(nn.Sequential(
                nn.Conv2d(in_channel, filter_num, (filter_size, 1),(2,1)),#(2,1)
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),

            ))
        
        self.layers_conv = nn.ModuleList(layers_conv)

        # PART2 , Cross Channel Fusion through Attention
        self.dropout = nn.Dropout(dropout)

        self.sa = SelfAttention(filter_num, sa_div)
        


        # PART 3 , Prediction 
        
        self.activation = nn.ReLU() 
        self.fc1 = nn.Linear(input_shape[3]*filter_num ,hidden_dim)

        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            2,
            bidirectional=False,
            dropout=0.15,
        )

        self.prediction = nn.Linear(hidden_dim ,number_class)


        


    def forward(self, x):
        # B ? L C
        # x = x.unsqueeze(1)
        
        
        for layer in self.layers_conv:
            x = layer(x)      


        batch, filter, length, channel = x.shape


        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )


       # print(refined.shape)

        x = refined.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.dropout(x)
        
        x = self.activation(self.fc1(x)) # B L F
        x = x.permute(1,0,2)

        outputs, h = self.rnn(x) # L B  F
        x = outputs[-1, :, :]
        y = self.prediction(x)    
        return y