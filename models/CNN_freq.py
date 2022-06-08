import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    
class CNN_Freq_Model(nn.Module):

    def __init__(self, input_shape, number_class):
        super(CNN_Freq_Model, self).__init__()

        self.dw1 = nn.Sequential(
            nn.Conv2d(input_shape[1], input_shape[1], 5, padding ="same",groups=input_shape[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(input_shape[1]),
            nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
        )
        
        self.dw2 = nn.Sequential(
            nn.Conv2d(input_shape[1], input_shape[1], 5, padding ="same",groups=input_shape[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(input_shape[1]),
            nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
        )
        
        self.dw3 = nn.Sequential(
            nn.Conv2d(input_shape[1], input_shape[1], 5, padding ="same",groups=input_shape[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(input_shape[1]),
            nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
        )
        
        self.se1 = SE_Block(input_shape[1], 1)

        self.pw1 = nn.Sequential(
            nn.Conv2d(input_shape[1], input_shape[1], 3, padding ="same",groups=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(input_shape[1])
        )

        self.dw4 = nn.Sequential(
            nn.Conv2d(input_shape[1], input_shape[1], 5, padding ="same",groups=input_shape[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(input_shape[1]),
            nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
        )
        
        self.se2 = SE_Block(input_shape[1], 1)

        self.pw2 = nn.Sequential(
            nn.Conv2d(input_shape[1], input_shape[1], 3, padding ="same",groups=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(input_shape[1])
        )
        
        shape = self.get_the_shape(input_shape)
        
        self.fc = nn.Linear(shape[2],1)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(shape[1]*shape[3],64)
        
        self.fc2 = nn.Linear(64,number_class)
    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.se1(x)
        x = self.pw1(x)
        x = self.dw4(x)
        x = self.se2(x)
        x = self.pw2(x)
        
        return x.shape
    
    def forward(self, x):
        #x = x.permute(0,3,1,2)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.se1(x)
        x = self.pw1(x)
        x = self.dw4(x)
        x = self.se2(x)
        x = self.pw2(x)
        x = self.activation(torch.squeeze(self.fc(x.permute(0,1,3,2)),3))

        x = self.flatten(x)

        
        x = self.activation(self.fc1(x))

        y = self.fc2(x)        

        return y
    
    