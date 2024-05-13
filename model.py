import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from gcn_layer import *
from maxvit import *
from tanet import TANet

INPLANES = 3

from mobile_net_v2 import mobile_net_v2, cat_net

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


class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.load_state_dict(torch.load('resnet18-5c106cde.pth'))
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)



class FineTunedResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=False)
        self.resnet101.load_state_dict(torch.load('resnet101-5d3b4d8f.pth'))
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
        self.fc3 = nn.Linear(512, 10)  

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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.dowansample = downsaple
        self.stride = stride

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dowansample is not None: 
            identity = self.dowansample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FPG(nn.Module):
    def __init__(self, block, layers, num_classes, gcn_num, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(FPG, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.block_name = block.__name__
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool3x3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.lagcn1 = LAGCN1_Layer(10 * 10, 1024, 1024, gcn_num, adj_op='combine')
        self.lagcn2 = LAGCN1_Layer(19 * 19, 1024, 1024, gcn_num, adj_op='combine')
        self.lagcn3 = LAGCN1_Layer(38 * 38, 512, 512, gcn_num, adj_op='combine')
        self.lagcn4 = LAGCN1_Layer(75 * 75, 256, 256, gcn_num, adj_op='combine')
        self.lagcn5 = LAGCN2_Layer(25, 1024, 1024)
        self.aesfc1 = nn.Linear(256 * 3 + 1024, num_classes)
        self.aesfc2 = nn.Linear(512 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, (H, W), mode='bilinear') + y

    #@torchsnooper.snoop()
    def forward(self, input):
        x, ratio_info = input[0], input[1]
        ratio_info = ratio_info.unsqueeze(1)
        x = self.conv1(x) # b,64,150,150
        c1 = self.maxpool(self.relu(self.bn1(x))) # b,64,75,75
        c2 = self.layer1(c1) #b,256,75,75
        c3 = self.layer2(c2) #b,512,38,38
        c4 = self.layer3(c3) #b,1024,19,19
        c5 = self.layer4(c4) #b,2048,10,10
        p5 = self.conv1x1(c5) #b,1024,10,10
        p5 = self.lagcn1([p5, ratio_info]) #b,1024,10,10
        c4 = self.lagcn2([c4, ratio_info]) #b,1024,19,19
        c3 = self.lagcn3([c3, ratio_info]) #b,512,38,38
        c2 = self.lagcn4([c2, ratio_info]) #b,256,75,75
        p4 = self._upsample_add(self.lateral1(p5), self.lateral2(c4)) #b,256,19,19
        p3 = self._upsample_add(p4, self.lateral3(c3)) #b,256,38,38
        p2 = self._upsample_add(p3, c2) #b,256,75,75

        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)

        o1_1 = self.avgpool(p5)
        o1_2 = self.avgpool(p4)
        o1_3 = self.avgpool(p3)
        o1_4 = self.avgpool(p2)
        o1 = torch.cat([o1_1,o1_2,o1_3,o1_4], dim=1)
        x = self.lagcn5(p5)

        o2 = self.avgpool(x)
        o1 = o1.view(o1.size(0), -1)
        o1 = self.aesfc1(o1)
        o1 = self.softmax(o1)
        o2 = o2.view(o2.size(0), -1)
        o2 = self.aesfc2(o2)
        o2 = self.softmax(o2)
        out = (o1 + o2) / 2
        # return o1, o2, out
        return out


def resnet50fpg(pretrained=True, num_classes=10, gcn_num=1, **kwargs):
    """Constructs a ResNet-50 HLA-GCN model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): number of the classificaiton bins
        gcn_num (int): number of the GCN block in the first LA-GCN module
    """
    model = FPG(Bottleneck, [3, 4, 6, 3], num_classes, gcn_num)
    if pretrained:
        pretrained_dict = torch.load('resnet50-19c8e357.pth')
        model_dict = model.state_dict()
        model_keys = model.state_dict().keys()
        for k in pretrained_dict:
            if k in model_keys:
                model_dict[k] = pretrained_dict[k]
        model.load_state_dict(model_dict)
    return model



class ReLIC(nn.Module):
    def __init__(self):
        super(ReLIC, self).__init__()
        base_model = cat_net()

        self.base_model = base_model
        for p in self.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(8,64)
        self.relu = nn.Tanh()
        self.fc1 = nn.Linear(64,2)
        # self.sm = nn.Softmax(dim=1) #multi classification
        self.sm = nn.Sigmoid() #binary classification


        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(3681, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1, x2 = self.base_model(x)
        x1_max = torch.max(x1,dim=1)[0].unsqueeze(1)
        x1_min = torch.min(x1,dim=1)[0].unsqueeze(1)
        x1_mean = torch.mean(x1,dim=1).unsqueeze(1)
        x1_std = torch.std(x1,dim=1).unsqueeze(1)
        x1_in = torch.cat([x1_max,x1_min,x1_mean,x1_std],1)

        x2_max = torch.max(x2,dim=1)[0].unsqueeze(1)
        x2_min = torch.min(x2,dim=1)[0].unsqueeze(1)
        x2_mean = torch.mean(x2,dim=1).unsqueeze(1)
        x2_std = torch.std(x2,dim=1).unsqueeze(1)
        x2_in = torch.cat([x2_max,x2_min,x2_mean,x2_std],1)

        x_in = torch.cat([x1_in,x2_in],1)
        x_out =self.sm(self.fc1(self.relu(self.fc(x_in))))

        x1 = x1 * torch.unsqueeze(x_out[:, 0], 1)
        x2 = x2 * torch.unsqueeze(x_out[:, 1], 1)
        x = torch.cat([x1,x2],1)
        x = self.head(x)

        return x



def create_model(model_type: str, drop_out: float):
    if model_type == 'nima':
        return NIMA()
    elif model_type == 'mlsp':
        return MLSP()
    elif model_type == 'resnet18':
        return FineTunedResNet18()
    elif model_type == 'resnet50':
        return FineTunedResNet50()
    elif model_type == 'resnet101':
        return FineTunedResNet101()
    elif model_type == 'slicemodel':
        return SliceModel()
    elif model_type == 'fpg':
        return resnet50fpg()
    elif model_type == 'relic':
        return ReLIC()
    elif model_type == 'MaxVIT':
        return max_vit_base_224(num_classes=10)
    elif model_type == 'tanet':
        return TANet()
    else:
        print('Not implemented!')




