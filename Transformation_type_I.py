import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type I.
Conv_3x1x3 -> Conv_3x3x3
'''

class Net(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.conv313 = nn.Conv3d(out_shape, out_shape, (3, 1, 3))
        self.conv_fuse=nn.Conv3d(in_shape, out_shape, 3)

    def forward(self, x, fuse=False):
        if fuse:
            # self.fuse_conv_multiscale()
            return self.conv_fuse(x)
        else:
            return self.conv313(x)

    def fuse_conv_multiscale(self):
        self.conv_fuse.weight.data, self.conv_fuse.bias.data = \
            self.trans_conv_to_3x3x3(self.conv313)

    '''
    multi-scale 3x1x3 -> 3x3x3
    '''
    def trans_conv_to_3x3x3(self, conv):
        # 填充 3x3x1 卷积的权重
        shape=conv.weight.shape
        if shape[2:]==[3,3,1]:  # 1x3x3
            weight = F.pad(conv.weight.data, (1, 1, 0, 0, 0, 0))  # 填充
        elif shape[2:]==[3,1,3]:  # 3x1x3
            weight = F.pad(conv.weight.data, (0, 0, 1, 1, 0, 0))  # 填充
        elif shape[2:]==[1,3,3]:  # 1x3x3
            weight = F.pad(conv.weight.data, (0, 0, 0, 0, 1, 1))  # 填充
        else: weight = conv.weight
        # 合并偏置
        bias = (conv.bias.data if conv.bias is not None else 0)

        return weight, bias


feature_map=torch.randn(1,64,80,80,80)
net=Net(64,64)
# 重参数化之前的结果
net.eval()
out1=net(feature_map)
# 重参数化之后的结果
net.fuse_conv_multiscale()
out2=net(feature_map, fuse=True)

# 验证前后的结果是否相同
print("out1:", out1[0, 30, 40, 40:50, 40:50])
print("out2:", out2[0, 30, 40, 40:50, 40:50])
print("out1-out2:", (out1-out2)[0, 30, 40, 40:50, 40:50])
print("difference:", ((out2-out1)**2).sum().item())
