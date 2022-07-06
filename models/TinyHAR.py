import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

class SelfAttention_interaction(nn.Module):
    """

    """
    def __init__(self, n_channels):
        super(SelfAttention_interaction, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):

        # 输入尺寸是 batch  sensor_channel feature_dim
        #print(x.shape)

        f, g, h = self.query(x), self.key(x), self.value(x)
        
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)

        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        # 输出是 batch  sensor_channel feature_dim 1 
        return o.permute(0, 2, 1).contiguous()



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


class Transformer_interaction(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dim_head=16, mlp_dim=16, dropout = 0.):
        super(Transformer_interaction,self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):


        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class Identity(nn.Module):
    def __init__(self, n_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

crosschannel_interaction = {"attn":SelfAttention_interaction,
                            "transformer": Transformer_interaction,
                            "identity": Identity}

class FilterWeighted_Aggregation(nn.Module):
    """

    """
    def __init__(self, n_channels):
        super(FilterWeighted_Aggregation, self).__init__()
        self.value_projection = nn.Linear(n_channels, n_channels)
        self.value_activation = nn.ReLU() 
        
        self.weight_projection = nn.Linear(n_channels, n_channels)
        self.weighs_activation = nn.Tanh() 
        self.softmatx = nn.Softmax(dim=1)        

        
        
    def forward(self, x):
        
        # 输入是 batch  sensor_channel feature_dim


        weights = self.weighs_activation(self.weight_projection(x))
        weights = self.softmatx(weights)
        
        values  = self.value_activation(self.value_projection(x))

        values  = torch.mul(values, weights)
        # 返回是 batch feature_dim
        return torch.sum(values,dim=1)

class NaiveWeighted_Aggregation(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(NaiveWeighted_Aggregation, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):

        # 输入是 batch  sensor_channel feature_dim
        #   B C F

        out = self.fc(x).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        return context


class FC(nn.Module):

    def __init__(self, channel_in, channel_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(channel_in ,channel_out)

    def forward(self, x):
        x = self.fc(x)
        return(x)


crosschannel_aggregation = {"filter": FilterWeighted_Aggregation,
                            "naive" : NaiveWeighted_Aggregation,
                            "FC" : FC}



class temporal_GRU(nn.Module):
    """

    """
    def __init__(self, filter_num):
        super(temporal_GRU, self).__init__()
        self.rnn = nn.GRU(
            filter_num,
            filter_num,
            1,
            bidirectional=False,
            dropout=0.15,
            batch_first = True
        )
    def forward(self, x):
        # Batch length Filter
        outputs, h = self.rnn(x)
        return outputs


class temporal_LSTM(nn.Module):
    """

    """
    def __init__(self, filter_num):
        super(temporal_LSTM, self).__init__()
        self.lstm = nn.LSTM(filter_num, 
                            filter_num, 
                            batch_first =True)
    def forward(self, x):
        # Batch length Filter
        outputs, h = self.lstm(x)
        return outputs



temporal_interaction = {"gru": temporal_GRU,
                        "lstm": temporal_LSTM,
                        "attn"   :SelfAttention_interaction,
                        "transformer": Transformer_interaction,
                        "identity" : Identity}


temmporal_aggregation = {"filter": FilterWeighted_Aggregation,
                         "naive" : NaiveWeighted_Aggregation,
                         "FC" : FC,
                         "identiry":Identity}






class TinyHAR_Model(nn.Module):
    def __init__(
        self,
        input_shape ,
        number_class , 

        filter_num,
        nb_conv_layers                 = 4,        
        filter_size                    = 5,
        
        cross_channel_interaction_type = "attn",    # attn  transformer  identity
        cross_channel_aggregation_type = "filter",  # filter  naive  FC
        temporal_info_interaction_type = "gru",     # gru  lstm  attn  transformer  identity
        temporal_info_aggregation_type = "FC",      # naive  filter  FC 

        dropout = 0.2,

        activation = "ReLU",

    ):
        super(TinyHAR_Model, self).__init__()
        
        
        self.cross_channel_interaction_type = cross_channel_interaction_type
        self.cross_channel_aggregation_type = cross_channel_aggregation_type
        self.temporal_info_interaction_type = temporal_info_interaction_type
        self.temporal_info_aggregation_type = temporal_info_aggregation_type
        
        
        """
        PART 1 , ============= Channel wise Feature Extraction =============================        
        输入的格式为  Batch, filter_num, length, Sensor_channel        
        输出格式为为  Batch, filter_num, downsampling_length, Sensor_channel
        """

        layers_conv = []
        for i in range(nb_conv_layers):
            if i == 0:
                in_channel = input_shape[1]
            else:
                in_channel = filter_num
    
            layers_conv.append(nn.Sequential(
                nn.Conv2d(in_channel, filter_num, (filter_size, 1),(2,1)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),

            ))
        self.layers_conv = nn.ModuleList(layers_conv)
        # 这是给最后时间维度 vectorize的时候用的
        downsampling_length = self.get_the_shape(input_shape)        
        

        """
        PART2 , ================ Cross Channel interaction  =================================
        这里可供选择的  attn   transformer  itentity
        输出格式为  Batch, filter_num, downsampling_length, Sensor_channel
        
        """

        self.channel_interaction = crosschannel_interaction[cross_channel_interaction_type](filter_num)
        # 这里还是 B F C L  需要permute++++++++++++++

        

        """
        PART3 , =============== Cross Channel Fusion  ====================================
        这里可供选择的  filter   naive  FC

        输出格式为  Batch, downsampling_length, filter_num
        """
        if cross_channel_aggregation_type == "FC":
            # 这里需要reshape为 B L C*F++++++++++++++
            self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](input_shape[3]*filter_num,filter_num)

        else:
            # 这里需要沿着时间轴走
            self.channel_fusion = crosschannel_aggregation[cross_channel_aggregation_type](filter_num)
            # --> B F L
            # 需要reshape++++++++++++++++++++++++++++++

            
        # BLF
        self.activation = nn.ReLU() 


        """
        PART4  , ============= Temporal information Extraction =========================
        这里可供选择的  gru lstm attn transformer   identity

        输出格式为  Batch, downsampling_length, filter_num
        """
        
        # ++++++++++++ 这里需要讨论
        self.temporal_interaction = temporal_interaction[temporal_info_interaction_type](filter_num)
        
        
        """
        PART 5 , =================== Temporal information Aggregation ================


        输出格式为  Batch, downsampling_length, filter_num
        """        

        self.dropout = nn.Dropout(dropout)
        
        if temporal_info_aggregation_type == "FC":
            self.flatten = nn.Flatten()
            self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](downsampling_length*filter_num,filter_num)
        else:
            self.temporal_fusion = temmporal_aggregation[temporal_info_aggregation_type](filter_num)
            
        #--> B F

        # PART 6 , ==================== Prediction ==============================
        self.prediction = nn.Linear(filter_num ,number_class)

    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)

        for layer in self.layers_conv:
            x = layer(x)    

        return x.shape[2]
        


    def forward(self, x):
        # B F L C   
        for layer in self.layers_conv:
            x = layer(x)

        x = x.permute(0,3,2,1) 
        # ------->  B x C x L* x F*       



        """ =============== cross channel interaction ==============="""
        x = torch.cat(
            [self.channel_interaction(x[:, :, t, :]).unsqueeze(3) for t in range(x.shape[2])],
            dim=-1,
        )
        # ------->  B x C x F* x L* 
        
        x = self.dropout(x)

        """=============== cross channel fusion ==============="""
        
        if self.cross_channel_aggregation_type == "FC":
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = self.activation(self.channel_fusion(x)) # B L C
        else:
            x = torch.cat(
                [self.channel_fusion(x[:, :, :, t]).unsqueeze(2) for t in range(x.shape[3])],
                dim=-1,
            )
            x = x.permute(0,2,1)
            x = self.activation(x)
        # ------->  B x L* x F*
            
            
        """cross temporal interaction """
        x = self.temporal_interaction(x)


        
        """cross temporal fusion """
        if self.temporal_info_aggregation_type == "FC":
            x = self.flatten(x)
            x = self.activation(self.temporal_fusion(x)) # B L C
        else:
            x = self.temporal_fusion(x)
        
        

        y = self.prediction(x)
        return y










