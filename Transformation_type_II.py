import torch
from torch import nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type II:
Conv-BN --> Conv
'''

class Net(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.conv1=nn.Conv3d(in_shape, out_shape, 3)
        self.BN=nn.BatchNorm3d(out_shape)
        self.fuse_conv=nn.Conv3d(in_shape, out_shape, 3)

    def forward(self, x, fuse=False):
        if fuse is False:
            return self.BN(self.conv1(x))
        else:
            # self.fuse_conv_BN()
            return self.fuse_conv(x)

    def fuse_conv_BN(self):
        self.fuse_conv.weight.data, self.fuse_conv.bias.data=self.transII_BN(self.conv1, self.BN)

    '''
    3x3x3-BN -> 3x3x3
    '''
    def transII_BN(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        gamma = bn.weight

        weight = conv.weight * (gamma / std).reshape(-1, 1, 1, 1, 1)

        if conv.bias is not None:
            bias = (gamma / std * conv.bias) - (gamma / std * bn.running_mean) + bn.bias
        else:
            bias = bn.bias - (gamma / std * bn.running_mean)

        return weight, bias


feature_map=torch.randn(1,64,80,80,80)
net=Net(64,64)
# 不要重参数化的结果
net.eval()
out1=net(feature_map)
# 重参数化之后的结果
net.fuse_conv_BN()
out2=net(feature_map, fuse=True)
# 验证前后的结果是否相同
print("out1:", out1[0, 30, 40, 40:50, 40:50])
print("out2:", out2[0, 30, 40, 40:50, 40:50])
