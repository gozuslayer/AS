-- Deep Q-learning with experience Replay
require 'nn'
require 'torch'
--importing and initialisation de l'environnement
require 'rlenvs'

local CartPole = require 'rlenvs.CartPole'
local env = CartPole()

--Initialisation du réseau de neurone pour apprendre Q 
-- réseau simple avec une couche caché
local n = 10

local mlp = nn.Sequential()
mlp:add(nn.Linear(4,n))
mlp:add(nn.Linear(n,2))
local criterion= nn.MSECriterion()



-- Initialisation du replay memory
 
--Capacity N
local N = 10
local memory_replay = torch.Tensor(N,5):zero() -- capacity X (actual_state, action, reward, futur_state,terminated)

-- Initialisation de Q
local StateShape = env:getStateSpace().shape[1]
local Q = torch.randn(StateShape,StateShape)

-- paramètre d'apprentissage
local NbrEpisode = 1000
local T = 50
local sizeMiniBatch = N/10
local gamma = 0.1
local learning_rate = 0.001


for episode=1,NbrEpisode do

	--Initialisation sequence
	env:start()
	--
	for t=1,T do
		-- select action with probability p (cartpole 2 actions)
		local p = 0.5
		local action = torch.bernoulli(p)

		--Execute action and observe result
		local observation = env:step(action) --(reward,new_state,terminal)

		--Store Observation in replay memory
		local idx = torch.random(1,N)
		memory_replay[idx][1] = previous_state
		memory_replay[idx][2] = action
		memory_replay[idx][3] = observation[1]
		memory_replay[idx][4] = observation[2]
		memory_replay[idx][5] = observation[3]

		--MiniBatch
		local idx_miniBatch = torch.randperm(N)[{{1,sizeMiniBatch}}]
		local MiniBatch = memory_replay[{idx_miniBatch}]

		--setting yj...
		local Y = torch.Tensor(sizeMiniBatch,1)
		for i=1,sizeMiniBatch do
			if (MiniBatch[i][5]==true) then
				Y[i]=MiniBatch[i][3]
			else 
				local QValues = torch.Tensor(2)
				local input1 = torch.cat(MiniBatch[i][5] , 0) --concat futur_state,action
				local input2 = torch.cat(MiniBatch[i][5] , 1) 
				QValues[1] = mlp:forward(input1)
				QValues[2] = mlp:forward(input2)
				Y[i]=MiniBatch[i][3] + gamma*torch.max(QValues)
			end
		end

		--learning
		mlp:zeroGradParameters()
		output=mlp:forward(MiniBatch)	
		delta=criterion:backward(output,Y)
		mlp:backward(MiniBatch,delta)
		mlp:updateParameters(learning_rate) 
	end
end