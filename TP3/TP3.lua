require 'nn'
require 'gnuplot'
require 'tools'

-- generation gaussienne (mu = {mu_1, mu_2,..}, sigma = {sig_1,sig_2,...})
gen_gauss = function(nbpoints, mu,sigma)
	local d = #mu
	local X = torch.randn(nbpoints,d)
	for i = 1,d do  X[{{},i}]=X[{{},i}]*sigma[i]+mu[i] end
	return X
end

-- deux gaussiennes et les labels
gen_bigauss = function(nbpoints, mu1, mu2, sigma1,sigma2)
	X=torch.cat(gen_gauss(nbpoints/2,mu1,sigma1),gen_gauss(nbpoints/2,mu2,sigma2),1)
	Y = torch.cat(torch.ones(nbpoints/2,1),-1*torch.ones(nbpoints/2,1))
	idx = torch.randperm(nbpoints):long()
	return X:index(1,idx),Y:index(1,idx)
end 

-- Creation des donnees
X, Y = gen_bigauss(200,{1,1},{-1,-1},{1,1},{1,1})

--Creation du network
mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 5; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
mlp:add(nn.Tanh())

--Parametre d'apprentissage
criterion = nn.MSECriterion()
local learning_rate= 1e-3 
local maxEpoch= 500
local all_losses={}
local timer = torch.Timer()
local all_losses ={}

for iteration = 1,maxEpoch do

  local loss=0
  local x,y,out,delta
  for j=1,xtrain:size(1) do
    x=xtrain[j]
    y=ytrain[j]
    out=model:forward(x)
    loss=loss+criterion:forward(out,y)
  end  
  loss=loss/N 
  all_losses[iteration]=loss

  --stochastique gradient
  local idx = math.random(Xtrain:size(1));
  x=Xtrain[idx]
  y=Ytrain[idx] 



  out=mlp:forward(x)
  delta=criterion:backward(out,y)
  mlp:backward(x,delta) 
  mlp:updateParameters(learning_rate)

  gnuplot.plot(torch.Tensor(all_losses))   

end

local function accuracy(y,out)
  local count = 0
  local Matrix = torch.cmul(y,out)
  for i =1, #y do
    if torch.sign(Matrix)>0 then
      count = count + 1
    end
  end
  local result = count/#y
  return result
end
