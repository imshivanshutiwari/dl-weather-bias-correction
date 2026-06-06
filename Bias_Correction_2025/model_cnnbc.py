import torch
from torch import nn

'''
class CNNBC(nn.Module):

    def __init__(self):
        super(CNNBC, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv3d(1, 1, (9,9,9), padding='same'), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(nn.Conv3d(1, 1, (3, 3,3), padding='same'), nn.ReLU(), nn.Dropout(0.1))
        self.conv3 = nn.Conv3d(1, 1, (5, 5,5), padding='same')
        
        self.conv4 = nn.Sequential(nn.Conv3d(1, 1, (9, 9,9), padding='same'), nn.ReLU(), nn.Dropout(0.2))
        self.conv5 = nn.Sequential(nn.Conv3d(1, 1, (3, 3,3), padding='same'), nn.ReLU(), nn.Dropout(0.1))
        self.conv6 = nn.Conv3d(1, 1, (5, 5,5), padding='same')
        
        self.conv7 = nn.Sequential(nn.Conv3d(1, 1, (10,1,1), stride=(10,1,1)), nn.ReLU())
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4((x+x3)/2)
        x5 = self.conv5((x1+x4)/2)
        x6 = self.conv6((x2+x5)/2)
        out = self.conv7(x6)
        
        return out
'''


class CNNBC(nn.Module):

    def __init__(self):
        super(CNNBC, self).__init__()
        
        self.conv0_1 = nn.Sequential(nn.Conv3d(1, 1, (10,15,15), padding='same'), nn.ReLU(), nn.Dropout(0.4))
        self.conv1 = nn.Sequential(nn.Conv3d(1, 1, (9,9,9), padding='same'), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(nn.Conv3d(1, 1, (3, 3,3), padding='same'), nn.ReLU(), nn.Dropout(0.1))
        self.conv3 = nn.Conv3d(1, 1, (5, 5,5), padding='same')
        
        self.conv3_4 = nn.Sequential(nn.Conv3d(1, 1, (10,15,15), padding='same'), nn.ReLU(), nn.Dropout(0.4))
        self.conv4 = nn.Sequential(nn.Conv3d(1, 1, (9, 9,9), padding='same'), nn.ReLU(), nn.Dropout(0.2))
        self.conv5 = nn.Sequential(nn.Conv3d(1, 1, (3, 3,3), padding='same'), nn.ReLU(), nn.Dropout(0.1))
        self.conv6 = nn.Conv3d(1, 1, (5, 5,5), padding='same')
        
        self.conv6_7 = nn.Sequential(nn.Conv3d(1, 1, (10,15,15), padding='same'), nn.ReLU(), nn.Dropout(0.4))
        self.conv7 = nn.Sequential(nn.Conv3d(1, 1, (9, 9,9), padding='same'), nn.ReLU(), nn.Dropout(0.2))
        self.conv8 = nn.Sequential(nn.Conv3d(1, 1, (3, 3,3), padding='same'), nn.ReLU(), nn.Dropout(0.1))
        self.conv9 = nn.Conv3d(1, 1, (5, 5,5), padding='same')
        
        self.conv_f = nn.Sequential(nn.Conv3d(1, 1, (10,1,1), stride=(10,1,1)), nn.ReLU())
        
    def forward(self, x):
        x0_1=self.conv0_1(x)
        x1 = self.conv1(x0_1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3_4=self.conv3_4(x3)
        x4 = self.conv4((x+x3_4)/2)
        x5 = self.conv5((x1+x4)/2)
        x6 = self.conv6((x2+x5)/2)
        x6_7=self.conv6_7(x6)
        x7 = self.conv7((x3+x6_7)/2)
        x8 = self.conv8((x4+x7)/2)
        x9 = self.conv9((x5+x8)/2)
        
        out = self.conv_f(x9)
        
        return out
        
