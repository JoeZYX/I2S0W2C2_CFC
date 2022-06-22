import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== SelfAttention_crosschannel_interaction=======================
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

class SelfAttention_crosschannel_interaction(nn.Module):
    """

    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention_crosschannel_interaction, self).__init__()

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

# ===================== SelfAttention_Aggregation======================= 
class SelfAttention_Aggregation(nn.Module):
    """

    """
    def __init__(self, n_channels):
        super(SelfAttention_Aggregation, self).__init__()
        self.value_projection = nn.Linear(n_channels, n_channels)
        self.weight_projection = nn.Linear(n_channels, n_channels)
        self.softmatx = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # B C F
        weights = self.weight_projection(x)
        weights = self.softmatx(weights)
        values  = self.value_projection(x)
        values  = torch.mul(values, weights)
        return torch.sum(values,dim=1).unsqueeze(2)

        
        
   # ===================== SelfAttention_Aggregation=======================      
class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        # L B F
        out = self.fc(x).squeeze(2)
        # L B
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context
    
class CFC_V4_Model(nn.Module):
    def __init__(
        self,
        input_shape ,
        number_class , 
        filter_num = 32,
        filter_size = 5,
        nb_conv_layers = 4,
        dropout = 0.2,
        hidden_dim = 16,
        activation = "ReLU",
        sa_div= 1,
    ):
        super(CFC_V4_Model, self).__init__()
        
        # PART 1 , ============= Channel wise Feature Extraction =============================
        
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

        # PART2 , ================ Cross Channel interaction  =================================
        self.dropout = nn.Dropout(dropout)
        self.channel_interaction = SelfAttention_crosschannel_interaction(filter_num, sa_div)
        

        # PART3 , =============== Cross Channel Fusion  ====================================

        
        self.channel_fusion = SelfAttention_Aggregation(filter_num)
    
        # PART4  , ============= Temporal information Extraction =========================

        self.rnn = nn.GRU(
            filter_num,
            filter_num,
            2,
            bidirectional=False,
            dropout=0.15,
        )
        # PART 5 , =================== Temporal information Aggregation ================
        self.temporal_fusion = TemporalAttention(filter_num)

        # PART 6 , ==================== Prediction ==============================
        self.prediction = nn.Linear(filter_num ,number_class)


        


    def forward(self, x):
        # B F L C   F==1 or F==Nds+1
        for layer in self.layers_conv:
            x = layer(x)      

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)

        x = torch.cat(
            [self.channel_interaction(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        # x B F C L

        x = x.permute(0,2,1,3)
        # refined B C F L

        x = torch.cat(
            [self.channel_fusion(x[:, :, :,t]) for t in range(x.shape[3])],
            dim=-1,
        )
        # refined B F L
        

        x = self.dropout(x)
        x = x.permute(2,0,1)
        x, h = self.rnn(x) # L B  F

        x = self.temporal_fusion(x)
        y = self.prediction(x)
        return y