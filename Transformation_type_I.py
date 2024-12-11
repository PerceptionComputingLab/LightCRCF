import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The validation of the equivalent transformation type I.
Conv_3x3x1, Conv_3x1x3, Conv_1x3x3 ---> Conv_3x3x3
'''

class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv331 = nn.Conv3d(in_channel, out_channel, (3, 3, 1), padding=(1, 1, 0))
        self.conv313 = nn.Conv3d(in_channel, out_channel, (3, 1, 3), padding=(1, 0, 1))
        self.conv133 = nn.Conv3d(in_channel, out_channel, (1, 3, 3), padding=(0, 1, 1))
        self.conv_fuse=nn.Conv3d(in_channel, out_channel, (3, 3, 3), padding=(1, 1, 1))

    def forward(self, x, test_type='conv313', fuse=False):
        if fuse:
            self.fuse_conv_multiscale(test_type)
            return self.conv_fuse(x)
        else:
            if test_type == 'conv331':
                return self.conv331(x)
            elif test_type == 'conv313':
                return self.conv313(x)
            elif test_type == 'conv133':
                return self.conv133(x)
            else:
                raise TypeError('test_type')

    def fuse_conv_multiscale(self, test_type='conv313'):
        if test_type == 'conv331':
            conv_fuse = self.conv331
        elif test_type == 'conv313':
            conv_fuse = self.conv313
        elif test_type == 'conv133':
            conv_fuse = self.conv133
        else:
            raise TypeError('test_type')
        self.conv_fuse.weight.data, self.conv_fuse.bias.data = \
            self.trans_conv_to_3x3x3(conv_fuse)

    def trans_conv_to_3x3x3(self, conv):
        # padding weight
        shape = list(conv.weight.shape[2:])
        if shape==[3,3,1]:  # 3x3x1
            weight = F.pad(conv.weight.data, (1, 1, 0, 0, 0, 0))
        elif shape==[3,1,3]:  # 3x1x3
            weight = F.pad(conv.weight.data, (0, 0, 1, 1, 0, 0))
        elif shape==[1,3,3]:  # 1x3x3
            weight = F.pad(conv.weight.data, (0, 0, 0, 0, 1, 1))
        else: weight = conv.weight
        # 合并偏置
        bias = (conv.bias.data if conv.bias is not None else 0)

        return weight, bias


feature_map=torch.randn(1,32,80,80,80)
net=Net(32,64)
net.eval()

test_type = 'conv331'  # 'conv331' 'conv313' 'conv133'

# 重参数化之前的结果
out1=net(feature_map, test_type=test_type)
# 重参数化之后的结果
out2=net(feature_map, test_type=test_type, fuse=True)

# 验证前后的结果是否相同
print("out1:", out1[0, 30, 40, 40:50, 40:50])
print("out2:", out2[0, 30, 40, 40:50, 40:50])
print("out1-out2:", (out1-out2)[0, 30, 40, 40:50, 40:50])
print("difference:", ((out2-out1)**2).sum().item())
print(out1.shape, out2.shape)
