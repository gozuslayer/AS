require 'nn'
require 'gnuplot'
require 'utils'

-- Creation des donnees
X, Y = gen_bigauss(2000,{1,1},{-1,-1},{1,1},{1,1})

--Spliting in train and test
Xtrain, Xtest, Ytrain, Ytest = splitTrainTest(X,Y,0.1)

--Creation du network
mlp = nn.Sequential();  
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
local all_train_accuracy={}
local all_test_accuracy={}

for iteration = 1,maxEpoch do

  local loss=0
  local x,y,out,delta
  for j=1,Xtrain:size(1) do
    x=Xtrain[j]
    y=Ytrain[j]
    loss=loss+criterion:forward(mlp:forward(x),y)
  end  
  loss=loss/Xtrain:size(1)
  all_losses[iteration]=loss

  mlp:zeroGradParameters()
  --stochastique gradient
  local idx = math.random(Xtrain:size(1));
  out=mlp:forward(Xtrain[idx])
  delta=criterion:backward(out,Ytrain[idx] )
  mlp:backward(Xtrain[idx],delta) 
  mlp:updateParameters(learning_rate)
  
  all_train_accuracy[iteration] = accuracy(Ytrain,mlp:forward(Xtrain))
  all_test_accuracy[iteration] = accuracy(Ytest,mlp:forward(Xtest))
  
  gnuplot.plot({'train',torch.Tensor(all_train_accuracy)},{'test',torch.Tensor(all_test_accuracy)}) 
  gnuplot.movelegend('left','top')
  gnuplot.title('Accuracy on Training and testing Set')
end

--print(all_train_accuracy)
--print(all_test_accuracy)