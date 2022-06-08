import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_CROSS_Model(nn.Module):

    def __init__(self, input_shape_t, input_shape_tf, number_class , number_filters_t=48, number_filters_tf=48):
        super(CNN_LSTM_CROSS_Model, self).__init__()
        
        # input_shape  L F
 
        # ---------------------------------------  
        self.dwt_1 = nn.Sequential(
            nn.Conv1d(input_shape_t[1], number_filters_t, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_t),

        )

        self.dwtf_1 = nn.Sequential(
            nn.Conv1d(input_shape_tf[1], number_filters_tf, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_tf),
        )

        # ---------------------------------------
        
        self.dwt_2 = nn.Sequential(
            nn.Conv1d(number_filters_t, number_filters_t, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_t),

        )

        self.dwtf_2 = nn.Sequential(
            nn.Conv1d(number_filters_tf, number_filters_tf, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_tf),
        )

        # ---------------------------------------
        self.dwt_3 = nn.Sequential(
            nn.Conv1d(number_filters_t, number_filters_t, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_t),

        )

        self.dwtf_3 = nn.Sequential(
            nn.Conv1d(number_filters_tf, number_filters_tf, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_tf),
        )


        # ---------------------------------------
        self.dwt_4 = nn.Sequential(
            nn.Conv1d(number_filters_t, number_filters_t, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_t),

        )

        self.dwtf_4 = nn.Sequential(
            nn.Conv1d(number_filters_tf, number_filters_tf, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters_tf),
        )

        # ---------------------------------------


        self.lstm_layers_t_1 = nn.LSTM(number_filters_t, number_filters_t, batch_first =True)
        self.lstm_layers_t_2 = nn.LSTM(number_filters_t, number_filters_t, batch_first =True)

        self.lstm_layers_tf_1 = nn.LSTM(number_filters_tf, number_filters_tf, batch_first =True)
        self.lstm_layers_tf_2 = nn.LSTM(number_filters_tf, number_filters_tf, batch_first =True)


        self.activation = nn.ReLU(inplace=True)
        #self.fc_t_tf = nn.Linear((number_filters_tf+number_filters_t), 64)
        #self.fc = nn.Linear(64, number_class)
        self.fct = nn.Linear(number_filters_t, number_class)
        self.fctf = nn.Linear(number_filters_tf, number_class)
        self.fc = nn.Linear(number_class,number_class)
    def forward(self, x_t, x_tf):

        batch_t, length_t, channel_t = x_t.shape
        x_t = x_t.permute(0,2,1)
        x_t = self.dwt_1(x_t)
        x_t = self.dwt_2(x_t)
        x_t = self.dwt_3(x_t)
        x_t = self.dwt_4(x_t) # B F L
        x_t = x_t.permute(0, 2, 1)
        x_t, _ = self.lstm_layers_t_1(x_t)
        x_t, _ = self.lstm_layers_t_2(x_t)
        x_t = x_t[:,-1,:]

		
        x_tf = x_tf[:,0,:,:]
        batch_tf, Freq_tf, length_tf = x_tf.shape
        x_tf = self.dwtf_1(x_tf)
        x_tf = self.dwtf_2(x_tf)
        x_tf = self.dwtf_3(x_tf)
        x_tf = self.dwtf_4(x_tf) # B F L
        x_tf = x_tf.permute(0, 2, 1)
        x_tf, _ = self.lstm_layers_tf_1(x_tf)
        x_tf, _ = self.lstm_layers_tf_2(x_tf)
        x_tf = x_tf[:,-1,:]

        #x = torch.cat((x_t, x_tf), 1)

        #x = self.activation(self.fc_t_tf(x))
        #x = self.fc(x)
        x_t = self.fct(x_t)
        x_tf = self.fctf(x_tf)
        x = self.fc(self.activation(x_t+x_tf))
        return x
    
    