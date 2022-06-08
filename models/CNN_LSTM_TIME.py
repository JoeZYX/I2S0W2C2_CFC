import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_TIME_Model(nn.Module):

    def __init__(self, input_shape, number_class , number_filters=76):
        super(CNN_LSTM_TIME_Model, self).__init__()
        
        # input_shape  L F
   
        self.dw1 = nn.Sequential(
            nn.Conv1d(input_shape[1], number_filters, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters),

        )
        
        self.dw2 = nn.Sequential(
            nn.Conv1d(number_filters, number_filters, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters),

        )
        
        self.dw3 = nn.Sequential(
            nn.Conv1d(number_filters, number_filters, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters),

        )
        
        self.dw4 = nn.Sequential(
            nn.Conv1d(number_filters, number_filters, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_filters)
        )

        self.lstm_layers_1 = nn.LSTM(number_filters, number_filters, batch_first =True)
        self.lstm_layers_2 = nn.LSTM(number_filters, number_filters, batch_first =True)
        
        self.fc = nn.Linear(number_filters, number_class)


    def forward(self, x):

        batch, length,channel = x.shape
        x = x.permute(0,2,1)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.dw4(x) # B F L
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_layers_1(x)
        x, _ = self.lstm_layers_2(x)
        
        x = x[:,-1,:]
        
        x = self.fc(x)
        return x
    
    