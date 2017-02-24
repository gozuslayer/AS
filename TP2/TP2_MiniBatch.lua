require 'nn'
require 'gnuplot'
require 'utils'

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
	for i =1, y:size(1) do
		if y[i]*out[i]>0 then
			count = count + 1
		end
	end
	local result = count/y:size(1)
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
local maxEpoch=50
local all_losses={}
local size_minibatch = 10
local Blocs = torch.round(N/size_minibatch)
local timer = torch.Timer()


for iteration=1,maxEpoch do
    
    local loss=0  
    ---- calcul de la loss moyenne 
 	for j=1,xtrain:size(1) do
		x=xtrain[j]
    	y=ytrain[j]
    	out=model:forward(x)
    	loss=loss+criterion:forward(out,y)
    end  
    loss=loss/N	
    all_losses[iteration]=loss  --stockage de la loss moyenne (pour dessin)

    -- version gradient minibatch

    --get index of each minibatch
    local idx = torch.randperm(N)
    local index_MiniBatch_X = idx:chunk(Blocs,1)
    local index_MiniBatch_Y = idx:chunk(Blocs,1)

    
    for i=1,Blocs do

        model:zeroGradParameters()
        MiniBatch_X = xtrain:index(1,index_MiniBatch_X[i]:long())
        MiniBatch_Y = ytrain:index(1,index_MiniBatch_Y[i]:long())
            	
        out=model:forward(MiniBatch_X)
        delta=criterion:backward(out,MiniBatch_Y)
        model:backward(MiniBatch_X,delta) 
        model:updateParameters(learning_rate)  
    end


    -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
    --plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
    --gnuplot.plot(torch.Tensor(all_losses)) 
end

local function prediction(xtest,ytest)
	output = model:forward(xtest)
	return accuracy(ytest,output)
	end

print(timer:time().real)
print(prediction(xtest,ytest))


