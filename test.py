 
import torch
import numpy
import time

if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

x = 8000
y = 1000

###CPU
start_time = time.time()
a = torch.ones(x,x)
for _ in range(y):
    a += a
elapsed_time = time.time() - start_time

print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(x,x).cuda()
for _ in range(y):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)

