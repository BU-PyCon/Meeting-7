import numpy as np
import os.path
from random import random
from matplotlib.pyplot import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# This function is not necessary for the ANN creation, it is merely
# for diagnostic purposes to periodically check how the ANN is doing
# by testing it on a selection of random data. It will print out,
# overall, how well the ANN is doing and show 20 randomly selected
# spectra and their results after being fed through the ANN.
def testANN():
    correct = 0
    printSample = 0
    # Loop through the testing input data (distinct from the
    # training input data) and feed that through the network.
    # Then, if we meet some random condition (which allows us
    # to print out different results each time we test) then
    # print out the ANN results for that testing sample.
    for i in range(-1, -(numbOfFiles-numbOfSamples), -1):
        test = net.activate(inp[i]) # Resultant vector from the ANN
        compare = out[i]            # Expected vector for the input spectra
        # If we meet the random condition and haven't already printed out
        # 20 random test results, let's print this particular spectra out.
        if printSample < 20 and random() < 0.05:
            prob = sigmoid(test)/sum(sigmoid(test)) # Sigmoid and normalize our vector
            pos = np.argsort(-prob)[:3] # Sort it and choose the three highest components
            # Now print out information about how this test spectra compares, indicating
            # the three most likely spectra as chosen by the ANN, their percentages, and
            # the correct answer.
            print('Testing Result', -i, '-',
                  'Expected:', slist[np.where(compare == max(compare))[0][0]],
                  flush = True, end = '')
            print('  Computed: ',
                  slist[pos[0]], ' ',
                  str(round(prob[pos[0]]*100,1))[:4], '%   ',
                  slist[pos[1]], ' ',
                  str(round(prob[pos[1]]*100,1))[:4], '%   ',
                  slist[pos[2]], ' ',
                  str(round(prob[pos[2]]*100,1))[:4], '%   ',
                  sep = '', flush = True)
            printSample += 1
        # Check if the ANN was right for this spectra and increase the correct
        # counter if it was.
        equality = np.all((test == max(test)) == (compare == max(compare)))
        if (equality == True): correct += 1
    # Print out how many of the test spectra it got right at this stage.
    print('Testing Data Results:', correct, '/', numbOfFiles-numbOfSamples, flush = True)
    return correct

# This is run once and when the ANN is finally done being trained. It will apply
# the ANN to the testing portion of the data and see how many the ANN was able
# to get correct. This will return the number of correct classifications, the
# total number that were correct in each spectral class and the total number of
# spectra in each class.
def finalTest():
    correct = 0
    categoryCorrect = [0]*len(slist)
    categoryAll = np.zeros(len(slist))
    for i in range(-1, -(numbOfFiles-numbOfSamples), -1):
        test = net.activate(inp[i])
        compare = out[i]
        categoryAll[compare == 1] += 1
        equality = np.all((test == max(test)) == (compare == max(compare)))
        if (equality == True):
            correct += 1
            prob = sigmoid(test)/sum(sigmoid(test))
            pos = np.argsort(-prob)[0]
            categoryCorrect[pos] += 1
    print('Testing Data Results:', correct, '/', numbOfFiles-numbOfSamples, flush = True)
    return correct, categoryCorrect, categoryAll

# A simple sigmoid function defined for normalizing values
def sigmoid(x):
    return 1/(1+np.exp(-x))

print('Start...', flush = True)

##
# DEFINE VARIABLES
#

PATH = 'PUT_IN_PATH_TO_ANN_DATA'
FILE = 'ANN_Data.txt'
slist = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T']
numbOfFiles = 8000    # The total number of input spectra to use in creating this ANN
numbOfSamples = 6000  # The total number of input spectra to use solely for training
numbOfTraining = 100  # The number of times to train the spectra on the training sample

inp = np.array([])
out = np.array([])

##
# ANN SETUP
#

# Define a network with 4500 input nodes, one for each spectrum resolution element,
# a single hidden layer with 100 nodes, and an output layer with 9 nodes (one for
# each of the possible spectral types. This represents a probabilistic neural network.
# This network uses bias nodes and applies a sigmoid function along the way.
net = buildNetwork(4500, 100, 9, bias = True, hiddenclass = SigmoidLayer)
ds = SupervisedDataSet(4500, 9)

# Read in the input data file. This is a list where on each line is the flux measured
# at each wavelength and the last value in the line is that spectrums known spectral type.
# this is all extracted 
f = 0; c = 0
with open(PATH+FILE) as infile:
    for line in infile:
        line = line.strip()
        stype = line[-1]
        line = np.asarray(line[0:-3].split(', ')).astype(np.float)
        line = line[:4500]
        # Only use spectra which are not just noise
        if (np.mean(line) >= 4):
            # Remove any values in the spectra which are obviously wrong by zeroing them
            line[np.abs(line) > 1000] = 0
            if (f < numbOfSamples):
                # Add the current spectra and known spectral type as a 9-element
                # vector. The spectra is scaled down by 1000.
                ds.addSample( line/1000, [(stype == s)*1 for s in slist])
            # Put the spectra and spectral type into our own arrays for later
            inp = np.append(inp, line/1000)
            out = np.append(out, [(stype == s)*1 for s in slist])
            f += 1  # The number of spectra used so far
        c += 1      # The number of spectra read in so far
        if (c % 100 == 0): print(c, f, flush = True)
        if (f >= numbOfFiles):
            break

##
# ANN TRAINING
#

print('Training ANN...', flush = True)

out = out.reshape((f,9))
inp = inp.reshape((f,4500))
trainer = BackpropTrainer(net, ds, learningrate = 0.05)
err = []
# Begin training the requrested number of times. Every tenth
# training session, test the ANN to see how it is coming along.
for i in range(numbOfTraining):
    err.append(trainer.train())
    print('Training Session', i, flush = True)
    if (i % 10 == 0):
        print('Training Error:', err[-1], flush = True)
        testANN()

# Now the training is done, we need to do a final test of
# the data and analyze how well the ANN did at the end.
correct, categoryCorrect, categoryAll = finalTest()
print(correct)
# Convert categoryCorrect to a percentage
for i in range(len(categoryCorrect)):
    print(categoryCorrect[i], categoryAll[i])
    if (categoryAll[i] > 0):
        categoryCorrect[i] /= categoryAll[i] * 100
    else:
        categoryCorrect[i] = np.nan


##
# OUTPUT
#

filename = 'output_'
i = 1
while os.path.isfile(filename + str(i) + '.txt'):
    i += 1
filename += str(i)+'.txt'
print('Printing to ' + filename, flush = True)

# Print information concerning the total accuracy of the ANN
file = open(filename,'w')
file.write('Final accuracy: ' + str(correct) + '/' + str(numbOfFiles - numbOfSamples) + '\n')
file.write('Accuracy per category: ')
for i in categoryCorrect: file.write(str(i)+', ')
file.write('\n\n')

# Print out the ANN structure, just for fun
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


print('Done...', flush = True)

