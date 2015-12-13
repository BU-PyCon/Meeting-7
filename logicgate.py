import numpy as np
import os.path
from matplotlib.pyplot import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

print('Start ...')

# Builds basic network with default set up. Two input nodes and a
# single output node.
net = buildNetwork(2, 5, 1, bias = True, hiddenclass = SigmoidLayer)

# Create the supervised data for training. We're telling the trainer
# that it will have two input and one output.
ds = SupervisedDataSet(2, 1)

# Put in the training data. Input first, then expected output.
# Change output to change gate
ds.addSample( (-1, -1), (-1,)) # Both False
ds.addSample( ( 1, -1), (-1,)) # First is True
ds.addSample( (-1,  1), (-1,)) # Second is True
ds.addSample( ( 1,  1), ( 1,)) # Both True

print('Training network ...')

# Now train using back-propagation
trainer = BackpropTrainer(net, ds, learningrate = 0.1)
err = [] # This will be an array of the error in the array as it evolves
for i in range(1000):
    err.append(trainer.train())

print('Outputing results ...')

# Now that its trained, let's try and run it on new data and
# print out what the ANN gets. We'll also print the last error
# value from the error array.
print('ANN Results')
print('-----------')
print('(F, F) =', net.activate((-1,-1)))
print('(T, F) =', net.activate(( 1,-1)))
print('(F, T) =', net.activate((-1, 1)))
print('(T, T) =', net.activate(( 1, 1)))
print('Error in ANN =', err[-1])

# For pedagogical purposes, let's inspect the ANN structure.
# This will print out all the weights used between the layers.
print('\nANN Network Structure')
print('---------------------')
for mod in net.modules:
    for conn in net.connections[mod]:
        print('>>Layer ', mod.name, ' (', mod.dim, ') ', sep = '', end='')
        print('is connected to layer ', conn.outmod.name, ' (', conn.outmod.dim, ') with weights:', sep = '')
        print(np.array(conn.params).reshape(mod.dim,conn.outmod.dim))
        print('')

# Plot up the error array so we can see how the ANN accuracy
# evolved over time.
figure()
plot(err)
gca().set_yscale('log')
show(block=False)

# And let's also print out the ANN structure to a file.
filename = 'output_'
i = 1
while os.path.isfile(filename + str(i) + '.txt'):
    i += 1
filename += str(i)+'.txt'

file = open(filename,'w')
file.write('Final accuracy: ' + str(800) + '/1000\n\n')

for mod in net.modules:
    for conn in net.connections[mod]:
        file.write(mod.name + ' -> ' + conn.outmod.name + ', ' + str(mod.dim) + ' x ' + str(conn.outmod.dim) + '\n')
        x = np.array(conn.params).reshape(mod.dim,conn.outmod.dim)
        for i in range(mod.dim):
            for j in range(conn.outmod.dim):
                file.write(str(x[i,j]) + ' ')
            file.write('\n')
        file.write('\n')

file.close()

print('Done ...')
