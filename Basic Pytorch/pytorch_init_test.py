import torch

# Test cuda and GPU model - 5070 Needs at least 12.8 or 13.0
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# See if GPU is actually used
x = torch.rand(10000,10000,device='cuda')
y = torch.matmul(x,x)
print("Done on:", x.device)