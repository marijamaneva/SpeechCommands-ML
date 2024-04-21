# Script in which I trained the model with batch gradient descent and stochastic gradient descent 
# by trying also minibatches of different size

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



# training of the MLP model for 10 loops. In each loop it performs mini batch training
# using a learning rate of 1e-4, 20 steps per loop and a batch size of 20
# training and test accuracies are also calculated for each loop
m = Xtrain.shape[0]
# by adding 50 I try multilayers 
network = pvml.MLP([1600, 35])


train_accuracies = []
test_accuracies = []
# GRADIENT DESCENT
for epoch in range(20):
    network.train(Xtrain_meanvar, Ytrain, lr = 1e-4, steps = m//m, batch = m)
    predictions, logits = network.inference(Xtrain_meanvar)
    train_acc_meanvar = (predictions==Ytrain).mean()
    predictions, logits = network.inference(Xtest_meanvar)
    test_acc_meanvar = (predictions==Ytest).mean()
    print(f"Epoch{epoch}, train {train_acc_meanvar: .3f} test {test_acc_meanvar :.3f}" )
# the trained MLP model is saved to a file called "mlp.npz" 

    # Append accuracies to the lists
    train_accuracies.append(train_acc_meanvar)
    test_accuracies.append(test_acc_meanvar)

# Plotting the accuracies
epochs = range(1, 21)  # Assuming 20 epochs
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracies')
plt.legend()
plt.show()  
network.save("mlp-desc-grad.npz")


train_accuracies = []
test_accuracies = []
# STOCHASTIC GRADIENT
for epoch in range(20):
    network.train(Xtrain_meanvar, Ytrain, lr = 1e-4, steps = m//1, batch = 1)
    predictions, logits = network.inference(Xtrain_meanvar)
    train_acc_meanvar = (predictions==Ytrain).mean()
    predictions, logits = network.inference(Xtest_meanvar)
    test_acc_meanvar = (predictions==Ytest).mean()
    print(f"Epoch{epoch}, train {train_acc_meanvar: .3f} test {test_acc_meanvar :.3f}" )
    
    # Append accuracies to the lists
    train_accuracies.append(train_acc_meanvar)
    test_accuracies.append(test_acc_meanvar)

# Plotting the accuracies
epochs = range(1, 21)  # Assuming 20 epochs
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracies')
plt.legend()
plt.show()
    
# the trained MLP model is saved to a file called "mlp.npz"   
network.save("mlp-stoc-grad.npz")
