# ---- import models ---------------
from models.Attend import AttendDiscriminate
from models.SA_HAR import SA_HAR
from models.deepconvlstm import DeepConvLSTM
from models.deepconvlstm_attn import DeepConvLSTM_ATTN
from models.crossatten.model import Cross_TS,TSTransformer_Basic
from models.TinyHAR import TinyHAR_Model
# ------- import other packages ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml



class model_builder(nn.Module):
    """
    
    """
    def __init__(self, args, input_f_channel = None):
        super(model_builder, self).__init__()

        self.args = args
        if input_f_channel is None:
            f_in  = self.args.f_in
        else:
            f_in  = input_f_channel

        if self.args.wavelet_filtering and self.args.wavelet_filtering_regularization:
            print("Wavelet Filtering Regularization")
            shape      = (1, f_in, 1, 1)
            self.register_parameter('gamma' , nn.Parameter(torch.ones(shape)))
            # self.gamma = nn.Parameter(torch.ones(shape))
            #self.register_buff
		

        self.first_conv = args.first_conv
        #if self.first_conv:
        #    self.pre_conv = nn.Conv2d()??


        if self.args.model_type == "tinyhar":
            self.model  = TinyHAR_Model((1,f_in, self.args.input_length, self.args.c_in ), 
                                         self.args.num_classes,
                                         cross_channel_interaction_type = self.args.cross_channel_interaction_type,    # attn  transformer  identity
                                         cross_channel_aggregation_type = self.args.cross_channel_aggregation_type,  # filter  naive  FC
                                         temporal_info_interaction_type = self.args.temporal_info_interaction_type,     # gru  lstm  attn  transformer  identity
                                         temporal_info_aggregation_type = self.args.temporal_info_aggregation_type)    # naive  filter  FC )
            print("Build the TinyHAR model!")

        elif self.args.model_type == "attend":
            self.model  = AttendDiscriminate((1,f_in, self.args.input_length, self.args.c_in ), 
                                             self.args.num_classes,
                                             self.args.filter_scaling_factor)
            print("Build the AttendDiscriminate model!")

        elif self.args.model_type == "deepconvlstm_attn":
            self.model  = DeepConvLSTM_ATTN((1,f_in, self.args.input_length, self.args.c_in ), 
                                            self.args.num_classes,
                                            self.args.filter_scaling_factor)
            print("Build the deepconvlstm_attn model!")

        elif self.args.model_type == "deepconvlstm":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm"]
            self.model  = DeepConvLSTM((1,f_in, self.args.input_length, self.args.c_in ), 
                                       self.args.num_classes,
                                       self.args.filter_scaling_factor,
                                       config)
            print("Build the DeepConvLSTM model!")
        else:
            raise NotImplementedError



    def forward(self,x):
        #if self.first_conv ：
        #    x = self.pre_conv(x)

        if self.args.wavelet_filtering and self.args.wavelet_filtering_regularization:
            x = x * self.gamma
        y = self.model(x)
        return y









"""
from models.crossatten.model import Cross_TS,TSTransformer_Basic
from models.deepconvlstm import DeepConvLSTM
from models.SA_HAR import SA_HAR
from models.deepconvlstm_attn import DeepConvLSTM_ATTN
from models.Attend import AttendDiscriminate
from models.Attend_new import AttendDiscriminate_new
from models.CNN_freq import CNN_Freq_Model
from models.CNN_LSTM_FREQ import CNN_LSTM_FREQ_Model
from models.CNN_LSTM_TIME import CNN_LSTM_TIME_Model
from models.CNN_LSTM_TIME_FREQ import CNN_LSTM_CROSS_Model
from models.CFC import CFC_Model
from models.CFC_V1 import CFC_V1_Model
from models.CFC_V2 import CFC_V2_Model
from models.CFC_V3 import CFC_V3_Model
from models.CFC_V4 import CFC_V4_Model
from models.TinyHAR import TinyHAR_Model
        if self.args.model_type in ["time","freq","cross"]:
            model  = Cross_TS(self.args)
            print("Build the conv_TS model!")
        elif self.args.model_type == "basic":
            model  = TSTransformer_Basic(self.args)
            print("Build the basic TS model!")
        elif self.args.model_type == "deepconvlstm":
            model  = DeepConvLSTM(self.args.c_in, self.args.num_classes)
            print("Build the DeepConvLSTM model!")
        elif self.args.model_type == "sahar":
            model  = SA_HAR(self.args.c_in, self.args.input_length, self.args.num_classes)
            print("Build the SA_HAR model!")
        elif self.args.model_type == "deepconvlstm_attn":
            model  = DeepConvLSTM_ATTN(self.args.c_in, self.args.num_classes)
            print("Build the deepconvlstm_attn model!")
        elif self.args.model_type == "attend":
            model  = AttendDiscriminate(self.args.c_in, self.args.num_classes)
            print("Build the AttendDiscriminate model!")
        elif self.args.model_type == "attend_new":
            model  = AttendDiscriminate_new(self.args.c_in, self.args.num_classes)
            print("Build the AttendDiscriminate_new model!")
        elif self.args.model_type == "cnn_freq":
            model  = CNN_Freq_Model((1,self.args.c_in, self.args.sampling_freq, self.args.input_length ), self.args.num_classes)
            print("Build the CNN_Freq_Model model!")		
        elif self.args.model_type == "cnn_lstm_freq":
            model  = CNN_LSTM_FREQ_Model((self.args.input_length,self.args.sampling_freq ), self.args.num_classes)
            print("Build the CNN_LSTM_FREQ_Model model!")					
        elif self.args.model_type == "cnn_lstm_time":
            model  = CNN_LSTM_TIME_Model((self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CNN_LSTM_TIME_Model model!")		
        elif self.args.model_type == "cnn_lstm_cross":
            model  = CNN_LSTM_CROSS_Model((self.args.input_length, self.args.c_in ),(self.args.input_length,self.args.sampling_freq ), self.args.num_classes)
            print("Build the CNN_LSTM_CROSS_Model model!")
        elif self.args.model_type == "cfc":
            model  = CFC_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC model!")
        elif self.args.model_type == "cfcv1":
            model  = CFC_V1_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V1_Model model!")
        elif self.args.model_type == "cfcv2":
            model  = CFC_V2_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V2_Model model!")		
        elif self.args.model_type == "cfcv3":
            model  = CFC_V3_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V3_Model model!")
        elif self.args.model_type == "cfcv4":
            model  = CFC_V4_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V4_Model model!")


        elif self.args.model_type == "tinyhar":
            model  = TinyHAR_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), 
                                   self.args.num_classes,
                                   cross_channel_interaction_type = self.args.cross_channel_interaction_type,    # attn  transformer  identity
                                   cross_channel_aggregation_type = self.args.cross_channel_aggregation_type,  # filter  naive  FC
                                   temporal_info_interaction_type = self.args.temporal_info_interaction_type,     # gru  lstm  attn  transformer  identity
                                   temporal_info_aggregation_type = self.args.temporal_info_aggregation_type)    # naive  filter  FC )
            print("Build the TinyHAR_Model model!")	
			
        else:
            raise NotImplementedError
"""