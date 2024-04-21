# Script that loads training data, normalize it, loads a training neural network,
# performs inference on thest data and displays the confusion matrix 

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

data = np.load("speech-comands/test.npz")
# prints the shape of the test data sets
Xtest = data["arr_0"]
Ytest = data["arr_1"]
print(Xtest.shape, Ytest.shape)

# reshapes the Xtrain in a 2D array with dimensions 20x80
# not normalized spectrogram
spectrogram_no_nz = Xtrain[0,:].reshape(20,80)
# visualization of the spectrogram
#plt.imshow(spectrogram_no_nz)
#plt.colorbar()
#plt.show()

# MEAN/VARIANCE NORMALIZATION
mu = Xtrain.mean(0)            #mean
std = Xtrain.std(0)            #deviation standard 
Xtrain_meanvar = (Xtrain - mu) / std   #normalized Xtrain
Xtest_meanvar = (Xtest - mu) / std     #normalized Xtest 

# normalized spectrogram MEAN-VAR
spectrogram = Xtrain_meanvar[0,:].reshape(20,80)

# MIN-MAX SCALING NORMALIZATION
min_val = np.min(Xtrain)
max_val = np.max(Xtrain)
Xtrain_minmax = (Xtrain - min_val) / (max_val - min_val)
Xtest_minmax = (Xtest - min_val) / (max_val - min_val)

# normalized spectrogram MIN-MAX
spectrogram = Xtrain_minmax[0,:].reshape(20,80)

# L2 NORMALIZATION
epsilon = 1e-8
Xtrain_l2 = Xtrain / (np.linalg.norm(Xtrain, ord=2, axis=1, keepdims=True) + epsilon)
Xtest_l2 = Xtest / (np.linalg.norm(Xtest, ord=2, axis=1, keepdims=True) + epsilon)

# normalized spectrogram L2
spectrogram_l2 = Xtrain_l2[0, :].reshape(20, 80)


# to show the image of the spectrogram
# no frequency means silence, but when there are patterns it means that the speaker is speaking
# there are many power frequencies 

#plt.imshow(spectrogram)
#plt.colorbar()
#plt.show()

# function that takes a network object as input and displays the weights of the network as images.
# It visualizes the weights for each class (35 classes) in a subplot
def show_weights(network):
    plt.figure(figsize=(30, 10))
    w = network.weights[0]
    maxval = np.abs(w).max()
    for klass in range(35):
        plt.subplot(5, 7, klass+1)
        plt.imshow(w[:, klass].reshape(20, 80),
               cmap="seismic", vmin=-maxval, vmax=maxval)
        plt.title(words[klass])
    plt.colorbar()
    plt.show()

# function that calculates a confusion matrix given predicted and true labels
def make_confusion_matrix(predictions,labels):
    cmat = np.zeros((35,35))
    for i in range(predictions.size):
        cmat[labels[i], predictions[i]] += 1
    s = cmat.sum(1, keepdims = True)
    cmat /= s 
    return cmat

# function that displays the confusion matrix in a tabular format
def display_confusion_matrix(cmat):
    print(" " * 10, end ="" )
    for j in range(35):
        print(f"{words[j][:4]:4} ", end="")
    print()
    for i in range(35):
        print(f"{words[i]:10} ", end ="")
        for j in range(35):
            val = cmat[i,j]*100
            print(f"{val:4.1f}" , end="")
        print()

# function that displays the confusion matrix using a heatmap plot
def G_display_confusion_matrix(cmat):
    plt.imshow(cmat, cmap='Blues')
    plt.xticks(range(35), words, rotation=90)
    plt.yticks(range(35), words)
    for i in range(35):
        for j in range(35):
            val = cmat[i, j]*100
            plt.text(j, i, "%4.1f" % val, ha='center', va='center', size=6)
    plt.tight_layout()
    plt.show()

#------------------------------------------------
m = Xtrain.shape[0]
network = pvml.MLP.load("mlp1.npz")
w = network.weights[0]

#show_weights(network) 
predictions, logits = network.inference(Xtest_meanvar)
cmat = make_confusion_matrix(predictions,Ytest)
display_confusion_matrix(cmat)
G_display_confusion_matrix(cmat)
show_weights(network)

#More concentration in the low frequencies 
#------------------------------------------------
network2 = pvml.MLP.load("mlp2.npz")
w2 = network2.weights[0]

predictions2, logits = network2.inference(Xtest_minmax)
cmat = make_confusion_matrix(predictions2,Ytest)
display_confusion_matrix(cmat)
G_display_confusion_matrix(cmat)

#------------------------------------------------

network3 = pvml.MLP.load("mlp3.npz")
w3 = network3.weights[0]

predictions3, logits = network3.inference(Xtest_l2)
cmat = make_confusion_matrix(predictions3,Ytest)
display_confusion_matrix(cmat)
G_display_confusion_matrix(cmat)


