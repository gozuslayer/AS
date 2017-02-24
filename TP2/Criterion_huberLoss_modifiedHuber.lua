require 'nn'

--Implementation de critere pour des descente stochastique
--A rajouter des boucles pour descente en batch


--Critere de Smotth
local HuberLoss, parent = torch.class(’HuberLoss’, ’nn.Criterion’) -- heritage en torch

function HuberLoss:__init(delta=1) 
	
	self.delta = delta
	self.gradInput = torch.Tensor()
	self.output = 0
	end

function HuberLoss:forward(input, target) 

	return self:updateOutput(input, target)
	end

function HuberLoss:updateOutput(input, target) 

	if torch.abs(input-target)<1 then
		self.output = ( torch.pow((input-target),2) )/2
	else 
		self.output = self.delta*torch.abs(input-target)-self.delta/2
	end

	return self.output
	end

function HuberLoss:backward(input, target) 
	
	return self:updateGradInput(input, target)

	end

function HuberLoss:updateGradInput(input, target) 

	if torch.abs(input-target)<1 then
		self.gradInput = input*(input-target)
	else
		self.gradInput = self.delta
	end

	return self.gradInput
	end



local ModifiedHuber, parent = torch.class(’ModifiedHuber’, ’nn.Criterion’) 

function ModifiedHuber:__init() 
	
	self.gradInput = torch.Tensor()
	self.output = 0
	end

function ModifiedHuber:forward(input, target) 
	return self:updateOutput(input, target)
	end

function ModifiedHuber:updateOutput(input, target) 
	if (1-(input*target))<= 2 then
		X = torch.Tensor(2)
		X[1]=0
		X[2]=(1-(input*target))
		self.output = torch.max(X)*torch.max(X)
	else 
		self.output = -4*(input*target)
	end

	return self.output
	end

function ModifiedHuber:backward(input, target) 
	return self:updateGradInput(input, target)
	end
	
function ModifiedHuber:updateGradInput(input, target) -- a completer
	if (1-(input*target))<= 2 then

	return self.gradInput
	end

