require 'nn'

local ReQU, parent = torch.class('nn.ReQU','nn.Module')

function ReQU:__init()
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()
end

function ReQU:updateOutput(input)
	self.output = torch.cmax(input,0)
	self.output:pow(2)
	return self.output
end



function ReQU:updateGradInput(input,gradOutput)
	self.gradInput = torch.cmax(2*input,0)
	self.gradInput:cmul(gradOutput)
	return self.gradInput
end

