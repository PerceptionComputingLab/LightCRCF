import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type IV.
Concat(Conv_3x3x3, Conv_3x3x3, Conv_3x3x3) ---> Conv_3x3x3
'''

class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1=nn.Conv3d(in_channel, out_channel, 3, padding=1)
        self.conv2=nn.Conv3d(in_channel, out_channel, 3, padding=1)
        self.conv3=nn.Conv3d(in_channel, out_channel, 3, padding=1)
        self.conv_fuse=nn.Conv3d(in_channel, out_channel * 3, 3, padding=1)

    def forward(self, x, fuse=False):
        if fuse:
            self.fuse_concate()
            return self.conv_fuse(x)
        else:
            o1=self.conv1(x)
            o2=self.conv2(x)
            o3=self.conv3(x)
            return torch.cat([o1, o2, o3], dim=1)

    def fuse_concate(self):
        self.conv_fuse.weight.data, self.conv_fuse.bias.data = self.transIV_concat(
            [self.conv1.weight.data, self.conv2.weight.data, self.conv3.weight.data],
            [self.conv1.bias.data, self.conv2.bias.data, self.conv3.bias.data]
        )

    def transIV_concat(self, kernels, biases):
        return torch.cat(kernels, dim=0), torch.cat(biases)



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
