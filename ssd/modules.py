import torch.nn as nn
import torch


#Add a class for Concat module here:
class Concatenate(nn.Module):
    def __init__(self, channels):
        super(Concatenate, self).__init__()
        self.conv = nn.Conv2d(2*channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        rgb_x, lp_x = x[0], x[1]
        result = torch.cat((rgb_x, lp_x), dim=1)
        result = self.conv(result)
        return [result, lp_x]


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        # x.size() = (batch_size, chanenel, height, width)
        # L2Norm
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        #weight.size() = (512) -> (1,512,1,1)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights*x


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

