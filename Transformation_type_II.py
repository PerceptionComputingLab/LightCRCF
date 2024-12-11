import torch
from torch import nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type II:
Conv-BN ---> Conv
'''

class Net(nn.Module):
    def __init__(self, in_channel, out_channle):
        super().__init__()
        self.conv1=nn.Conv3d(in_channel, out_channle, 3, padding=1, bias=True)
        self.BN=nn.BatchNorm3d(out_channle)

        # 初始化BN中值
        self.BN.weight.data = torch.randn(out_channle).data
        self.BN.bias.data = torch.randn(out_channle).data
        self.BN.running_mean.data = torch.zeros(out_channle).data
        self.BN.running_var.data = torch.ones(out_channle).data

        self.fuse_conv=nn.Conv3d(in_channel, out_channle, 3, padding=1, bias=True)

    def forward(self, x, fuse=False):
        if fuse is False:
            return self.BN(self.conv1(x))
        else:
            self.fuse_conv_BN()
            return self.fuse_conv(x)

    def fuse_conv_BN(self):
        self.fuse_conv.weight.data, self.fuse_conv.bias.data=self.transI_BN(self.conv1, self.BN)

    '''
    3x3x3-BN -> 3x3x3
    '''
    def transI_BN(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        gamma = bn.weight

        weight = conv.weight * (gamma / std).reshape(-1, 1, 1, 1, 1)

        if conv.bias is not None:
            bias = (gamma / std * conv.bias) - (gamma / std * bn.running_mean) + bn.bias
        else:
            bias = bn.bias - (gamma / std * bn.running_mean)

        return weight, bias


feature_map=torch.randn(1,32,80,80,80)
net=Net(32,64)
net.eval()

# 重参数化之前的结果
out1=net(feature_map)
# 重参数化之后的结果
out2=net(feature_map, fuse=True)
# 验证前后的结果是否相同
print("out1:", out1[0, 30, 40, 40:50, 40:50])
print("out2:", out2[0, 30, 40, 40:50, 40:50])
print("out1-out2:", (out1-out2)[0, 30, 40, 40:50, 40:50])
print("difference:", ((out2-out1)**2).sum().item())
print(out1.shape, out2.shape)
