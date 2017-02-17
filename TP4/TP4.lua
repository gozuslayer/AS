require 'nn'

local MyModule, parent = torch.class('MyModule','nn.Module')

function MyModule:__init()
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()
end

function MyModule:updateOutput(input)
	
	for i=1,#input do
		if (input[i]>0) then
			self.output[i] = input[i]*input[i]
		end 
	end
	return self.output
end



function MyModule:updateGradInput(input,gradOutput)
	for i=1,#input do
		self.gradInput[i] = torch.max(0,2*input[i])*gradOutput
	end
	return self.gradInput
end


local Trans = MyModule()

local xs=torch.range(-1,1,10)
print (#xs)


print('forward')
print(Trans:forward(xs))

print('backward')
print(Trans:backward(xs,1))

