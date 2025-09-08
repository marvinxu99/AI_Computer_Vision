# https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch
import numpy as np

# Tensor init - directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Tensor init - from a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Tensor init - from another tensor
x_ones = torch.ones_like(x_data)            # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor Attributes
# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
