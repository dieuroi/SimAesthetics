import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

INPLANES = 3

from mobile_net_v2 import mobile_net_v2

class Conv2dBN(nn.Module):
    """
    CONV_BN_RELU
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBN, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))

class InceptionBlock(nn.Module):
    """
    InceptionBlock
    """

    def __init__(self, in_channels: object, out_channels: object) -> object:
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            Conv2dBN(in_channels, out_channels, kernel_size=1)#N*C*H*W
        )
        self.branch1x1 = Conv2dBN(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = Conv2dBN(in_channels, out_channels, kernel_size=1)
        self.branch3x3_1 = Conv2dBN(out_channels, out_channels//2, kernel_size=(1, 3), padding=(0, 1)) 
        self.branch3x3_2 = Conv2dBN(out_channels, out_channels//2, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        out1 = self.branch1(x)
        tmp = self.branch3x3(x)
        out1x1 = self.branch1x1(x)
        out3_1 = self.branch3x3_1(tmp)
        out3_2 = self.branch3x3_2(tmp)
        x = torch.cat((out1,out1x1,out3_1,out3_2),1)
        return F.adaptive_avg_pool2d(x, (1,1))
        
        

class MLSP(nn.Module):
    """
    MLSP
    """
    def __init__(self, in_channels=10048, out_channels=10, fc_sizes=[2048,1024,256,10], dropout_rates=[0.25,0.25,0.5,0]) -> object:
        super(MLSP, self).__init__()
        self.features = [None] * 11
        self.inception = models.inception_v3(pretrained=True)
        # self.inception_block = InceptionBlock(in_channels, 1024)
        # do BN for all except the last
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, fc_sizes[0]),
            nn.ReLU(True),
            nn.BatchNorm1d(fc_sizes[0]),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(True),
            nn.BatchNorm1d(fc_sizes[1]),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(fc_sizes[1], fc_sizes[2]),
            nn.ReLU(True),
            nn.BatchNorm1d(fc_sizes[2]),
            nn.Dropout(dropout_rates[3]),
            nn.Linear(fc_sizes[2], fc_sizes[3]),
            nn.ReLU(True),
            nn.Dropout(dropout_rates[3]),
            nn.Softmax(dim=1)
        )
        self.init_gap_layers()
        
        
    def init_gap_layers(self):
        def register_helper(idx, name):
            def copy (module, i, o):
                nonlocal self
                self.features[idx] = o
            getattr(self.inception, name).register_forward_hook(copy)
        Mixed_layers = ['5b', '5c', '5d', '6a', '6b', '6c', '6d', '6e', '7a', '7b', '7c']
        Mixed_layers = ['Mixed_{}'.format(ind) for ind in Mixed_layers]
        for i, name in enumerate(Mixed_layers):
            register_helper(i,name)
        
        
    #@torchsnooper.snoop()    
    def forward(self, x):
        self.inception.eval()
        out = []
        self.inception(x[0])
        for item in self.features:
            gap = torch.nn.AvgPool2d((item.shape[2], item.shape[3]))
            out.append(gap(item).squeeze(2).squeeze(2)) #F.adaptive_avg_pool2d(item, (1,1))
        self.features = [None] * 11
        x = torch.cat(out,1)
        x = self.fc_layers(x)
        return x



class NIMA(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMA, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x[0])
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)#?

    def forward(self, x):
        return self.resnet50(x)



class FineTunedResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet101(x)



class SliceModel(nn.Module):
    def __init__(self):
        super(SliceModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
      
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, d.num_classes)  

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.sigmoid(self.conv4(x))
        x = F.sigmoid(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 2048)  
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x




def create_model(model_type: str, drop_out: float):
    if model_type == 'nima':
        return NIMA()
    elif model_type == 'mlsp':
        return MLSP()
    elif model_type == 'resnet50':
        return FineTunedResNet50()
    elif model_type == 'resnet101':
        return FineTunedResNet101()
    elif model_type == 'slicemodel':
        return SliceModel()
    else:
        print('Not implemented!')




