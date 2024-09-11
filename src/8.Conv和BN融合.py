import time
import torch
import torch.nn as nn
import torchvision


# dummy: 占位符
class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):

        return x
    
def fuse(conv, bn):
    w        = conv.weight
    gamma    = bn.weight
    beta     = bn.bias
    mean     = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    # 权重融合
    # ((gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])): 广播
    w = w * ((gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1]))
    b = (b - mean) / var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias   = nn.Parameter(b)

    return fused_conv

def fuse_module(m):

    children = list(m.named_children()) # 获取m的所有子模块, 并存储为(name, child)的列表
    c  = None                           # c: 存储当前卷积层
    cn = None                           # cn: 存储当前卷积层的名字

    for name, child in children:

        # isinstance(object, classinfo): 检查一个对象是否是指定类型或其子类型的实例
        if isinstance(child, nn.BatchNorm2d):
            cb = fuse(c, child)
            m._modules[cn]   = cb            # 将融合后的卷积层替换原有的卷积层
            m._modules[name] = DummyModule() # 将原有的BN层替换为一个空的Module
            c = None
        elif isinstance(child, nn.Conv2d):
            c  = child
            cn = name
        else:
            fuse_module(child)               # 递归调用, 以处理子模块的子模块

def test_net(m):
    p = torch.randn([1, 3, 224, 224])

    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)

    s = time.time()
    f_output = m(p)
    print("Fused time: ", time.time() - s)
    print("Max abs diff: ", (o_output - f_output).abs().max().item())


def test_layer(m):
    p = torch.randn([1, 3, 224, 224])
    conv1 = m.conv1
    bn1 = m.bn1
    o_output = bn1(conv1(p))
    fusion = fuse(conv1, bn1)
    f_output = fusion(p)
    print(o_output[0][0][0][0].item())
    print(f_output[0][0][0][0].item())
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


print("============================")
print("Layer level test: ")
m = torchvision.models.resnet18(True)
m.eval()
test_layer(m)

print("Module level test: ")
m = torchvision.models.resnet18(True)
m.eval()
test_net(m)