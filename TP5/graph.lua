require 'nn'
require 'nngraph'

Dim = 2 
lamb = 0.1

--Duplicate inputs
inputs = {}
table.insert(inputs, nn.Identity()())
table.insert(inputs, nn.Identity()())

pred_input = inputs[1]
reg_input = inputs[2]

linear = nn.Linear(Dim, Dim)(inputs[1])

reg = nn.LookupTable(Dim, 1)(inputs[2])
penalty = nn.L1Penalty(lamb)(reg)
relu = nn.ReLU()(penalty)
cmul = nn.CMulTable()({ linear, relu })


linear_2 = nn.Linear(Dim, 1)(cmul)
decision = nn.Sigmoid()(linear_2)

outputs = {}
table.insert(outputs, decision)

g = nn.gModule(inputs, outputs)
graph.dot(g.fg, 'g')