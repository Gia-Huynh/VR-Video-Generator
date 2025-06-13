import torch
from torchvision.transforms import GaussianBlur

x = torch.randn(1, 3, 256, 256).cuda()
blur = GaussianBlur(kernel_size=5, sigma=1.0)
out = blur(x)
print(out.device)  # prints "cpu"
