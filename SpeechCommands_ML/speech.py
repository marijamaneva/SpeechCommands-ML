# Script that trains a MLP model with three different techniques of normalization

import numpy as np
import pvml
import matplotlib.pyplot as plt

# reads the file classes and the words are split and stored in words variable
words = open("speech-comands/classes.txt").read().split()
print(words)
# loads training data from the file train
data = np.load("speech-comands/train.npz")

# prints the shape of the training data sets
Xtrain = data["arr_0"]             #input features
Ytrain = data["arr_1"]             #corrisponding labels
print(Xtrain.shape, Ytrain.shape)

data = np.load("speech-comands/validation.npz")
# prints the shape of the test data sets
Xtest = data["arr_0"]
Ytest = data["arr_1"]
print(Xtest.shape, Ytest.shape)

# reshapes the Xtrain in a 2D array with dimensions 20x80
# not normalized spectrogram
spectrogram_no_nz = Xtrain[0,:].reshape(20,80)
plt.imshow(spectrogram_no_nz)
plt.colorbar()
plt.show()

# MEAN/VARIANCE NORMALIZATION
mu = Xtrain.mean(0)            #mean
std = Xtrain.std(0)            #deviation standard 
Xtrain_meanvar = (Xtrain - mu) / std   #normalized Xtrain
Xtest_meanvar = (Xtest - mu) / std     #normalized Xtest 

# normalized spectrogram MEAN-VAR
spectrogram_meanvar = Xtrain_meanvar[0,:].reshape(20,80)
plt.imshow(spectrogram_meanvar)
plt.colorbar() 
plt.show()


# MIN-MAX SCALING NORMALIZATION
min_val = np.min(Xtrain)
max_val = np.max(Xtrain)
Xtrain_minmax = (Xtrain - min_val) / (max_val - min_val)
Xtest_minmax = (Xtest - min_val) / (max_val - min_val)

# normalized spectrogram MIN-MAX
spectrogram_minmax = Xtrain_minmax[0,:].reshape(20,80)
plt.imshow(spectrogram_minmax)
plt.colorbar() 
plt.show()

# L2 NORMALIZATION
epsilon = 1e-8
Xtrain_l2 = Xtrain / (np.linalg.norm(Xtrain, ord=2, axis=1, keepdims=True) + epsilon)
Xtest_l2 = Xtest / (np.linalg.norm(Xtest, ord=2, axis=1, keepdims=True) + epsilon)

# normalized spectrogram L2
spectrogram_l2 = Xtrain_l2[0, :].reshape(20, 80)

# to show the image of the spectogram
# no frequency means silence, but when there are patterns it means that the speaker is speaking
# there are many power frequencies 

plt.imshow(spectrogram_l2)
plt.colorbar() 
plt.show()

# training of the MLP model for 10 loops. In each loop it performs mini batch training
# using a learning rate of 1e-4, 20 steps per loop and a batch size of 20
# training and test accuracies are also calculated for each loop
m = Xtrain.shape[0]
# by adding 50 I try multilayers 
network = pvml.MLP([1600, 35])
for epoch in range(10):
    network.train(Xtrain_meanvar, Ytrain, lr = 1e-4, steps = m//20, batch = 20)
    predictions, logits = network.inference(Xtrain_meanvar)
    train_acc_meanvar = (predictions==Ytrain).mean()
    predictions, logits = network.inference(Xtest_meanvar)
    test_acc_meanvar = (predictions==Ytest).mean()
    print(f"Epoch{epoch}, train {train_acc_meanvar: .3f} test {test_acc_meanvar :.3f}" )
# the trained MLP model is saved to a file called "mlp.npz"   
network.save("mlp1.npz")


#stochastic
# steps = m and batch = 1 
#gradient normale
#steps = 1 and batch = m 


for epoch in range(10):
    network.train(Xtrain_minmax, Ytrain, lr = 1e-4, steps = m//20, batch = 20)
    predictions, logits = network.inference(Xtrain_minmax)
    train_acc_minmax = (predictions==Ytrain).mean()
    predictions, logits = network.inference(Xtest_minmax)
    test_acc_minmax = (predictions==Ytest).mean()
    print(f"Epoch{epoch}, train {train_acc_minmax: .3f} test {test_acc_minmax :.3f}" )
# the trained MLP model is saved to a file called "mlp.npz"   
network.save("mlp2.npz")


for epoch in range(10):
    network.train(Xtrain_l2, Ytrain, lr = 1e-4, steps = m//20, batch = 20)
    predictions, logits = network.inference(Xtrain_l2)
    train_acc_l2 = (predictions==Ytrain).mean()
    predictions, logits = network.inference(Xtest_l2)
    test_acc_l2 = (predictions==Ytest).mean()
    print(f"Epoch{epoch}, train {train_acc_l2: .3f} test {test_acc_l2 :.3f}" )
# the trained MLP model is saved to a file called "mlp.npz"   
network.save("mlp3.npz")








