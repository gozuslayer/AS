require 'nn'

MyCriterion, Parent = torch.class('nn.MyCriterion', 'nn.Criterion')


function MyCriterion:__init()
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()
end 


function MyCriterion:forward(input, target)
	return self:updateOutput(input, target)
end


function MyCriterion:updateOutput(input, target)	
	self.output = torch.abs(input-target)
	return self.output
end

function MyCriterion:backward(input, target) 
	return self:updateGradInput(input, target)
end

function MyCriterion:updateGradInput(input, target) 
	self.gradInput = torch.sign(input-target) 
	return self.gradInput
end