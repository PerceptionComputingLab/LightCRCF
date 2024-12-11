import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type V.
Conv_3x3x3 + Conv_1x1x1 ---> Conv_3x3x3
'''

class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv3=nn.Conv3d(in_channel, out_channel, 3, padding=1)
        self.conv1=nn.Conv3d(out_channel, out_channel, 1, padding=0)
        self.conv_fuse=nn.Conv3d(in_channel, out_channel, 3, padding=1)

    def forward(self, x, fuse=False):
        if fuse:
            self.fuse_3x3x3_1x1x1()
            return self.conv_fuse(x)
        else:
            return self.conv1(self.conv3(x))

    def fuse_3x3x3_1x1x1(self):
        self.conv_fuse.weight.data, self.conv_fuse.bias.data = \
            self.transV_3x3x3_1x1x1(self.conv3, self.conv1)

    '''
    3x3x3 + 1x1x1 -> 3x3x3
    '''
    def transV_3x3x3_1x1x1(self, conv3, conv1):
        weight_conv1 = conv1.weight.squeeze()
        weight_conv3 = conv3.weight

        weight = torch.matmul(weight_conv1, weight_conv3.permute([2, 3, 4, 0, 1])).permute([3, 4, 0, 1, 2])

        bias_conv3 = conv3.bias
        bias = torch.matmul(weight_conv1, bias_conv3)
        if isinstance(conv1.bias, torch.Tensor):
            bias = bias + conv1.bias
        return weight, bias


feature_map=torch.randn(1,32,80,80,80)
net=Net(32,64)
net.eval()

# 重参数化之前的结果
out1=net(feature_map)
# 重参数化之后的结果
out2=net(feature_map,fuse=True)

# 验证前后的结果是否相同
print("out1:", out1[0, 30, 40, 40:50, 40:50])
print("out2:", out2[0, 30, 40, 40:50, 40:50])
print("out1-out2:", (out1-out2)[0, 30, 40, 40:50, 40:50])
print("difference:", ((out2-out1)**2).sum().item())
print(out1.shape, out2.shape)
