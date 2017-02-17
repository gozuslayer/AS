 require 'nn'
 require 'gnuplot'
 --require 'tools' 


 -- 1: Creation du jeux de données
 local DIMENSION=2 -- dimension d'entrée
 local n_points=1000 -- nombre de points d'apprentissage
  
   -- Tirage de deux gaussiennes
   local mean_positive=torch.Tensor(DIMENSION):fill(1); local var_positive=1.0
   local mean_negative=torch.Tensor(DIMENSION):fill(-1); local var_negative=1.0
   local xs=torch.Tensor(n_points,DIMENSION)
   local ys=torch.Tensor(n_points,1)
   for i=1,n_points/2 do  xs[i]:copy(torch.randn(DIMENSION)*var_positive+mean_positive); ys[i][1]=1 end
   for i=n_points/2+1,n_points do xs[i]:copy(torch.randn(DIMENSION)*var_negative+mean_negative); ys[i][1]=-1 end
   
 -- 2 : creation du modele
 -- TODO (Done)
 local model= nn.Linear(DIMENSION,1)
 local criterion= nn.MSECriterion()
 model:reset()
 
 
 local calculgradient = function(learning_rate) 
 
 -- 3 : Boucle d'apprentissage
 local learning_rate=0.1  
 local maxEpoch=100  
 local all_losses={}
 local N=10
 
 local idx = randperm(x:size(1))
 local timer = torch.Timer()
 
 print (X[1])
 for iteration=1,maxEpoch do
  ------ Mise à jour des paramètres du modèle
      ------ Evaluation de la loss moyenne 
    -- TODO
    local loss = 0
    ---- calcul de la loss moyenne 
    --stockage de la loss moyenne (pour dessin)
  
     -- version gradient stochastique
     -- TODO

    
  -- version gradient batch
  -- TODO
    for i=1,x:size(1)/10 do 
	batch = x:index(1,idx[
    	model:zeroGradParameters()	
    	output=model:forward(X[i])
    	loss=1/n_points*(criterion:forward(output,Y[i]))
    	delta=criterion:backward(output,Y[i])
    	model:backward(X[i],delta)
    	model:updateParameters(learning_rate)
    end
    
    for i=1,n_points do
        output=model:forward(x)
	loss2 = loss2 - 1/n_points*(criterion:forward(output,y))
    end

    all_losses[iteration]=loss2

    print (timer:time().real)
  -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
  --plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
  gnuplot.plot(torch.Tensor(all_losses)) 
end
 

 
