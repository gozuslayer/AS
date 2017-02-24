require 'nn'
require 'Linear'
require 'ReQU'
require 'Criterion'


local req = nn.ReQU()
local Linear = nn.MyLinear(10,1)
local criterion = nn.MyCriterion()

local ys=torch.Tensor({1})
local xs=torch.randn(10,1)
print (xs,ys)

print('ReQU')
print('forward pass')
print(req:forward(xs))

print('backward pass')
--print(req:backward(xs,))

print('Linear')
print('forward pass')
print(Linear:forward(xs))

print('backward pass')
--print(Linear:backward(xs,))

