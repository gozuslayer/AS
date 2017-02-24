require 'nn'
require 'nngraph'
require 'gnuplot'
load_mnist = require 'load_mnist'

local posSamples = 1
local negSamples = 7

local xsTrain, ysTrain = load_mnist.get_train(posSamples,negSamples)
local xsTest, ysTest = load_mnist.get_test(posSamples,negSamples)


-- 2 : creation du modele

local M=xsTrain:size(2)
local N=1


function GRU(input_size,rnn_size)
	local input = {}
	table.insert(inputs,nn.Identity()()) --corespond to sample X


end
-- GRU model
ht_1Input=nn.Identity()()
xInput=nn.Identity()()

Wr=nn.Linear(M,N)(xInput)
Wz=nn.Linear(M,N)(xInput)

Ur=nn.Linear(N,N)(ht_1Input)
Uz=nn.Linear(N,N)(ht_1Input)

rt=nn.CAddTable()({Ur,Wr})
rz=nn.CAddTable()({Uz,Wz})

rt_1=nn.Sigmoid()(rt)
zt=nn.Sigmoid()(rz)

rt_2=nn.CMulTable()({rt_1,ht_1Input})
rt_3=nn.Linear(N,N)(rt_2)


ht_tilde=nn.Tanh()(nn.CAddTable()({rt_3,nn.Linear(M,N)(xInput)}))

zt_tilde=nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(zt)),ht_1Input})

ht = nn.CAddTable()({zt_tilde,nn.CMulTable()({ht_tilde,zt})})

model=nn.gModule({ht_1Input,xInput},{ht})

-- Setting importants parameters

local ht_1=torch.Tensor(xsTrain:size(1),N):fill(0)

local criterion = nn.MSECriterion()

model:reset(0.1)
-- 3 : Boucle d'apprentissage

local learning_rate = 0.004
local maxEpoch = 800
local all_losses={}

for iteration=1,maxEpoch do


  model:zeroGradParameters()
  local output = model:forward({ht_1,xsTrain})

  ht_1=output

  local loss,b = criterion:forward(output, ysTrain)
  local delta = criterion:backward(output, ysTrain)

  model:backward(xsTrain,delta)
  model:updateParameters(learning_rate)
  
  all_losses[iteration] = loss

  
--	print(loss)
    gnuplot.figure(1) 
    gnuplot.plot(torch.Tensor(all_losses)) 
    -- gnuplot.figure(2)
    -- plot(xs,ys,model,100)
  
end