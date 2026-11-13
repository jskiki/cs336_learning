import torch

s1 = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s1 += torch.tensor(0.01, dtype=torch.float32)
print(s1)

s2 = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s2 += torch.tensor(0.01, dtype=torch.float16)
print(s2)

s3 = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s3 += torch.tensor(0.01, dtype=torch.float16)
print(s3)

s4 = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s4 += x.type(torch.float32)
print(s4)