# https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch
import numpy as np

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
else:
  print("cuda is not available.")


# Joining tensors 
# You can use torch.cat to concatenate a sequence of tensors along a given dimension. 
# See also torch.stack, another tensor joining op that is subtly different from torch.cat.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Multiplying tensors
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# In-place operations Operations that have a _ suffix are in-place. 
# For example: x.copy_(y), x.t_(), will change x.
print(tensor, "\n")
tensor.add_(5)
print(tensor)
