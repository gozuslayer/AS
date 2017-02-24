 require 'nn'
 require 'gnuplot'
 --require 'tools' 


 -- 1: Creation du jeux de données
 local DIMENSION=2 -- dimension d'entrée
 local n_points=4000 -- nombre de points d'apprentissage
  
   -- Tirage de deux gaussiennes
   local mean_positive=torch.Tensor(DIMENSION):fill(1); local var_positive=1.0
   local mean_negative=torch.Tensor(DIMENSION):fill(-1); local var_negative=1.0
   local xs=torch.Tensor(n_points,DIMENSION)
   local ys=torch.Tensor(n_points,1)
   for i=1,n_points/2 do  xs[i]:copy(torch.randn(DIMENSION)*var_positive+mean_positive); ys[i][1]=1 end
   for i=n_points/2+1,n_points do xs[i]:copy(torch.randn(DIMENSION)*var_negative+mean_negative); ys[i][1]=-1 end
   
local xtrain = xs
local ytrain = ys

local numberTrainingSample = xtrain:size(1)

 -- 2 : creation du modele
 -- TODO (Done)
 local model= nn.Linear(DIMENSION,1)
 local criterion= nn.MSECriterion()
 model:reset()
 
 

 
 -- 3 : Boucle d'apprentissage

 -- parametre d'apprentissage
 local learning_rate=0.001  
 local maxEpoch=1000
 local all_losses={}
 local timer = torch.Timer()

 
for iteration=1,maxEpoch do
  
  -- version gradient stochastique

	model:zeroGradParameters()
  local idx = math.random(xtrain:size(1));
  x=xtrain[idx]
  y=ytrain[idx] 
	output = model:forward(x)	
 	delta = criterion:backward(output,y)
 	model:backward(x,delta)	
 	model:updateParameters(learning_rate)

  local loss=0
  ---- calcul de la loss moyenne 
  -- store loss for visualisation
  for j=1,xtrain:size(1) do
      x=xtrain[j]
      y=ytrain[j]
      out=model:forward(x)
      loss=loss+criterion:forward(out,y)
  end 

  all_losses[iteration]=loss/numberTrainingSample
  print(loss)
  
  -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
  -- plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
  gnuplot.plot(torch.Tensor(all_losses)) 
end

print('time for training :',timer:time().real)

 

 
