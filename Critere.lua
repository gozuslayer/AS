require 'nn'

local MyCriterion, parent = torch.class('MyCriterion','nn.Criterion')

function MyCriterion:__init()
	self.gradInput = torch.Tensor()
	self.output = 0
end

function MyCriterion:forward(input,target)
	return self:updateOutput(input,target)
end

function MyCriterion:updateOutput(input,targe)
	self.output = torch.abs(input-target)
	return self.output
end

function MyCriterion:backward(input,target)
	return self:updateGradInput(input,target)
end

function MyCriterion:updateGradInput(input,target)
	if torch.abs(input-target)>0 do 
		self.gradInput = 1
	else
		self.gradInput = -1
	end
	return self.gradInput
end


