import torch
from torch.nn import functional as F
from mobile_net_v2 import *

def Attention(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

def MV2():
    model = mobile_net_v2()
    model = nn.Sequential(*list(model.children())[:-1])
    return model

class L5(nn.Module):
    def __init__(self):
        super(L5, self).__init__()
        back_model = MV2()
        self.base_model = back_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

        self.last_out_w = nn.Linear(365, 100)
        self.last_out_b = nn.Linear(365, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, x):
        res_last_out_w = self.last_out_w(x)
        res_last_out_b = self.last_out_b(x)
        param_out = {}
        param_out['res_last_out_w'] = res_last_out_w
        param_out['res_last_out_b'] = res_last_out_b
        return param_out

# L3
class TargetNet(nn.Module):

    def __init__(self):
        super(TargetNet, self).__init__()
        # L2
        self.fc1 = nn.Linear(365, 100)
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)
        self.bn1 = nn.BatchNorm1d(100).cuda()
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(1 - 0.5)

        self.relu7 = nn.PReLU()
        self.relu7.cuda()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, paras):

        q = self.fc1(x)
        # print(q.shape)
        q = self.bn1(q)
        q = self.relu1(q)
        q = self.drop1(q)

        self.lin = nn.Sequential(TargetFC(paras['res_last_out_w'], paras['res_last_out_b']))
        q = self.lin(q)
        q = self.softmax(q)
        return q

class TargetFC(nn.Module):
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        out = F.linear(input_, self.weight, self.bias)
        return out

class TANet(nn.Module):
    def __init__(self):
        super(TANet, self).__init__()
        self.res365_last = resnet365_backbone()
        self.hypernet = L1()

        # L3
        self.tygertnet = TargetNet()

        self.avg = nn.AdaptiveAvgPool2d((10, 1))
        self.avg_RGB = nn.AdaptiveAvgPool2d((12, 12))

        self.mobileNet = L5()
        self.softmax = nn.Softmax(dim=1)

        # L4
        self.head_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(20736, 10),
            nn.Softmax(dim=1)
        )

        # L6
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x_temp = self.avg_RGB(x)
        x_temp = Attention(x_temp)
        x_temp =x_temp.view(x_temp.size(0),-1)
        x_temp = self.head_rgb(x_temp)

        res365_last_out = self.res365_last(x)
        res365_last_out_weights = self.hypernet(res365_last_out)
        res365_last_out_weights_mul_out = self.tygertnet(res365_last_out, res365_last_out_weights)

        x2 = res365_last_out_weights_mul_out.unsqueeze(dim=2)
        x2 = self.avg(x2)
        x2 = x2.squeeze(dim=2)


        x1 = self.mobileNet(x)

        x = torch.cat([x1,x2,x_temp],1)
        x = self.head(x)

        return x
