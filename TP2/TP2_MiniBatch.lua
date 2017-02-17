require 'nn'
require 'gnuplot'
require 'tools'

--Load Data
local load_mnist = require 'load_mnist'
xtrain , ytrain = load_mnist.get_train(2,3)
xtest , ytest = load_mnist.get_test(2,3)
--xtrain,ytrain,xtest,ytest = load_mnist.get(2,3)

--gnuplot.imagesc(xtrain[1]:reshape(28,28))   --visualizing a sample

 -- 1: Creation du dataset (parameters)
local DIMENSION=xtrain:size(2) -- dimension 
local N=xtrain:size(1) -- nombre de sample


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

--- La différence entre une descente de gradient batch et stochastique et que l'on choist 
--plusieurs sample pour la descente de gradient (dans la mise a jour)

--- implementation de la descente de gradient mini batch sur un model linéaire
-- 2 : creation du modele
local model= nn.Linear(DIMENSION,1)
local criterion= nn.MSECriterion() 
 
 
 
-- 3 : Boucle d'apprentissage
local learning_rate= 1e-3 
local maxEpoch= 500
local all_losses={}
local size_minibatch = 10
local timer = torch.Timer()


for iteration=1,maxEpoch do
   	------ Evaluation de la loss moyenne 
    
    local loss=0  
    local x,y,out,delta
  
    ---- calcul de la loss moyenne 
 	for j=1,xtrain:size(1) do
		x=xtrain[j]
    	y=ytrain[j]
    	out=model:forward(x)
    	loss=loss+criterion:forward(out,y)
    end  
    loss=loss/N	
    all_losses[iteration]=loss  --stockage de la loss moyenne (pour dessin)

    -- version gradient stochastique
    model:zeroGradParameters()
    local idx = torch.randperm(N)
    local MiniBatch_X = torch.Tensor(size_minibatch)
    local MiniBatch_Y = torch.Tensor(size_minibatch)
    for i=1,size_minibatch do
        MiniBatch_X[i] = xtrain[idx[i]]
        MiniBatch_Y[i] = ytrain[idx[i]] 
            	
    out=model:forward(MiniBatch_X)
    loss=criterion:forward(out,MiniBatch_Y)
    delta=criterion:backward(out,MiniBatch_Y)
    model:backward(MiniBatch_X,delta) 
    model:updateParameters(learning_rate)  


    -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
    --plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
    gnuplot.plot(torch.Tensor(all_losses)) 
end

local function prediction(xtest,ytest)
	output = model:forward(xtest)
	return accuracy(ytest,output)
	end

print(timer:time().real)
print(prediction(xtest,ytest))


