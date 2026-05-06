import torch
import torchvision.models as tvm

m = tvm.mobilenet_v2()
x = torch.randn(2, 3, 256, 256)
out = x
for i, l in enumerate(m.features):
    out = l(out)
    print(i, out.shape)
