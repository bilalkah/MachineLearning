import torch
from torch import tensor

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z1 = torch.add(x,y)
z = x + y

# Substraction
z = x-y

#Division
z = torch.true_divide(x,y)

# Inplace operations
t = torch.zeros(3)
t.add_(x)
t += x # t = t + x

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple comparison
z = x > 0
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

# Matrix exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

# Element wise mult
z = x * y

# dot product
z = torch.dot(x,y)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2) # (batch,n,p)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x,dim=0)
print(x)
print(sum_x)
values, indices = torch.max(x,dim=0)
print(str(values)+""+str(indices))
values, indices = torch.min(x,dim=0)
print(str(values)+""+str(indices))
abs_x = torch.abs(x)
print(abs_x)
z = torch.argmax(x,dim=0) #argmin(x,dim_0)
print(z)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
print(z)
sorted_y, indices =  torch.sort(y,dim=0,descending=False)

z = torch.clamp(x,min=0,max=10)

x = torch.tensor([1,0,1,1,1],dtype=torch.bool)
z = torch.any(x)
z = torch.all(x) # False