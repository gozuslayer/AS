-- calculate accuracy between prediction and ground truth
function accuracy(y,out)

	local count = 0
	for i =1, y:size(1) do
		if y[i]*out[i]>0 then
			count = count + 1
		end
	end
	local result = count/y:size(1)
	return result
end

--split in train test with pourcent 1-s for train and s for test
function splitTrainTest(X,labels,s)
	

	--shuffling data and labels in same order
	local idx = torch.randperm(X:size(1))
	local X_shuffle = X:index(1,idx:long())
	local Y_shuffle = labels:index(1,idx:long())
	local Xtrain = X_shuffle[{{1,X:size(1)*(1-s)}}]
	local Xtest = X_shuffle[{{X:size(1)*(1-s)+1,X:size(1)}}]

	local Ytrain = Y_shuffle[{{1,X:size(1)*(1-s)}}]
	local Ytest = Y_shuffle[{{X:size(1)*(1-s) +1 , X:size(1)}}]
	return Xtrain, Xtest, Ytrain, Ytest

end

-- generation gaussienne (mu = {mu_1, mu_2,..}, sigma = {sig_1,sig_2,...})
function gen_gauss(nbpoints, mu,sigma)
	local d = #mu
	local X = torch.randn(nbpoints,d)
	for i = 1,d do  X[{{},i}]=X[{{},i}]*sigma[i]+mu[i] end
	return X
end

-- deux gaussiennes et les labels pour probleme XOR
function gen_bigauss(nbpoints, mu1, mu2, sigma1,sigma2)
	X = torch.cat(gen_gauss(nbpoints/2,mu1,sigma1),gen_gauss(nbpoints/2,mu2,sigma2),1)
	Y = torch.cat(torch.ones(nbpoints/2,1),-1*torch.ones(nbpoints/2,1),1)
	idx = torch.randperm(nbpoints):long()
	return X:index(1,idx),Y:index(1,idx)
end 
