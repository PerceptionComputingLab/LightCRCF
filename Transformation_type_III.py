import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type III.
Conv_1x1x1 + Conv_3x3x3 ---> Conv_3x3x3
'''

class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, 1, padding=1, bias=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, 3, padding=0, bias=True)
        self.conv_fuse = nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=True)

    def forward(self, x, fuse=False):
        if fuse:
            self.fuse_1x1x1_3x3x3()
            return self.conv_fuse(x)
        else:
            return self.conv2(self.conv1(x))

    def fuse_1x1x1_3x3x3(self):
        self.conv_fuse.weight.data, self.conv_fuse.bias.data = \
            self.transIII_1x1x1_3x3x3(self.conv1, self.conv2)

    '''
    1x1x1 + 3x3x3 -> 3x3x3
    '''
    def transIII_1x1x1_3x3x3(self, conv1, conv2):
        weight = F.conv3d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3, 4))
        b_hat = (conv2.weight.data * conv1.bias.data.reshape(1, -1, 1, 1, 1)).sum((1, 2, 3, 4))
        bias = b_hat + conv2.bias.data

        return weight, bias


feature_map=torch.randn(1,32,80,80,80)
net=Net(32,64)
net.eval()

# 重参数化之前的结果
out1 = net(feature_map)
# 重参数化之后的结果
out2 = net(feature_map, fuse=True)

# 验证前后的结果是否相同
print("out1:", out1[0, 30, 40, 40:50, 40:50])
print("out2:", out2[0, 30, 40, 40:50, 40:50])
print("out1-out2:", (out1-out2)[0, 30, 40, 40:50, 40:50])
print("difference:", ((out2-out1)**2).sum().item())
print(out1.shape, out2.shape)
