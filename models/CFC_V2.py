import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# -------------- Transformer Encoder -----------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 16, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AggregationAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(AggregationAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context

class Transformer(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dim_head=16, mlp_dim=16, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.aggregation = AggregationAttention(dim)
    def forward(self, x):
        # B F C 1
        size = x.size()
        # --> B C F
        x = x.view(*size[:2], -1).permute(0,2,1)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        # --- B C F 
        #x = self.aggregation(x.permute(1,0,2))
        return x.permute(0,2,1).unsqueeze(3)






    
class CFC_V2_Model(nn.Module):
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
        super(CFC_V2_Model, self).__init__()
        
        # PART 1 , Channel wise Feature Extraction
        
        layers_conv = []
        for i in range(nb_conv_layers):
        
            if i == 0:
                in_channel = input_shape[1]
            else:
                in_channel = filter_num
            if i%2==0:
                stride = 2
            else:
                stride = 2
            layers_conv.append(nn.Sequential(
                nn.Conv2d(in_channel, filter_num, (filter_size, 1),(stride,1)),#(2,1)
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),
            ))
        
        self.layers_conv = nn.ModuleList(layers_conv)

        # PART2 , Cross Channel Fusion through Attention
        self.dropout = nn.Dropout(dropout)
        #self.sa = SelfAttention(filter_num, sa_div)
        self.channel_aggregation = Transformer(dim=filter_num, depth=1, heads=4, dim_head=16, mlp_dim=filter_num, dropout = 0.)



        self.rnn = nn.GRU(
            filter_num*input_shape[3],
            hidden_dim,
            2,
            bidirectional=False,
            dropout=0.15,
        )
        
        #self.temporal_aggregation = AggregationAttention(hidden_dim)

#         # PART 3 , Prediction 
        self.prediction = nn.Linear(hidden_dim, number_class)


    def forward(self, x):
        # B F , L C  F =1 or F =filter * scale + 1

        
        for layer in self.layers_conv:
            x = layer(x)      
        # B filter_num  L  C

        batch, filter, length, channel = x.shape
        # 每次进去的都是 B filter_num C 1
        refined = torch.cat(
            [self.channel_aggregation(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        # B F C L


        x = refined.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # B L  F*C
        x = self.dropout(x)
        x = x.permute(1,0,2)
        outputs, h = self.rnn(x)

        # L B F
        x = outputs[-1, :, :]
        # B  F 

        y = self.prediction(x)    
        return y