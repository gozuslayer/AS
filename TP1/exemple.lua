 require 'nn'
 require 'gnuplot'

local Criterion = torch.class('nn.Criterion')

function Criterion:__init()
	self.gradInput = torch.Tensor()
	self.output = 0
end

function Criterion:updateOutput(input,target)
	self.output = output
end

function Criterion:forward(input,target)
	return self:updateOutput(input,target)
end

print(Criterion.gradInput)

