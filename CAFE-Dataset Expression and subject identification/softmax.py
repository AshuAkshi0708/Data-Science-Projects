# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:06:44 2019

@author: Ashya
"""
import numpy as np
import matplotlib.pyplot as plt
from PCA_images import load_train_test_images

#Plot beautifying parameters
plt.rcParams["lines.linewidth"] = 1.25
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["errorbar.capsize"] = 1.0

def softmax(z):
    '''Implements the softmax function for a vector containing w^Tx for the weights corresponding to each class
    Args
        z : Vector containing w^Tx
    Returns
        sm : Softmax vector containing the softmax probabilities for all the classes'''

    #Massaging because numpy vectors behave differently from numpy matrices
    if z.ndim != 1:
        sm = (np.exp(z)/ np.array([np.sum(np.exp(z),axis=1)]).T)
    else:
        sm= np.exp(z)/np.sum(np.exp(z))
    return sm

def onehotencoding(labels):
    '''
    Does one-hot encoding from the labels
    Args
        labels : List containing the labels
    Returns
        onehotCoded : Matrix containing the one-hot encoded values
    '''
    emotion_dict = {"ht": "happy",  "m": "maudlin","s": "surprise", "f": "fear", "a": "anger", "d": "disgust"}

    encode = dict((l, i) for i, l in enumerate(emotion_dict.values()))
    decode = dict((i, l) for i, l in enumerate(emotion_dict.values()))
    # integer encode input data
    int_encoded = [encode[l] for l in labels]
    # one hot encode
    onehotCoded = list()
    for value in int_encoded:
        letter = [0 for i in range(len(emotion_dict))]
        letter[value] = 1
        onehotCoded.append(letter)
    # invert encoding
    inverseCoded = decode[np.argmax(onehotCoded[0])]
    return np.array(onehotCoded)

def intencode(labels):
    '''
    Returns int-encoded labels i.e., the index where the one-hot encoding puts 1 in an array of zeros. Needed to compute confusion matrix
    Args
        labels : List containing the labels
    Returns
        int_encoded : List containing the int-encoded labels
    '''
    emotion_dict = {"ht": "happy",  "m": "maudlin","s": "surprise", "f": "fear", "a": "anger", "d": "disgust"}
    encode = dict((l, i) for i, l in enumerate(emotion_dict.values()))
    int_encoded = [encode[l] for l in labels]
    return int_encoded

def loss(h, y):
    '''Implements the softmax loss function. We take the average to "normalize" the losses. This does not affect the gradients
    Args
        h : w^Tx
        y : one-hot encoded labels
    Returns
        Softmax cost
        '''
    return -np.average(y * np.log(h))


def train_fit_gradient_descent(alpha,Iter,X,y_text,Xho,yho_text):
    '''
    Implements a batch gradient descent algorithm with early stopping based on holdout loss
    Args
        alpha : Learning rate
        Iter : Number of Epochs
        X : Training set
        y_text : Training labels
        Xho : Holdout set
        yho_text : Holdout labels
    Returns:
        theta : Best weights
        cost_array : Training cost for each epoch
        hocost_array : Holdout cost for each epoch
    '''
        #Get the one-hot encoded labels
    y = onehotencoding(y_text)
    yho = onehotencoding(yho_text)

    cost_array = np.zeros(Iter);
    hocost_array = np.zeros(Iter);
    curr_ho_cost = np.inf
    #Adding Intercept --> X becomes n*(d+1) size
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    Xho = np.concatenate((np.ones((Xho.shape[0], 1)), Xho), axis=1)
    #creating the parameter vector d+1 parameters
    theta = np.random.rand(X.shape[1],6);
    theta_sample = np.random.rand(X.shape[1],6)
    for i in range(0,Iter):
        cost = 0;
        hocost = 0;
        z = np.dot(X, theta)
        h = softmax(z)
        #Compute gradient
        gradient = np.dot(X.T, (h - y))
        theta_sample -= alpha * gradient
        hocost += loss(softmax(np.dot(Xho,theta_sample)),yho)
        hocost_array[i] = hocost
        #Early stopping
        if (hocost) < curr_ho_cost:
            theta = theta_sample;
            curr_ho_cost = hocost;
        cost+= loss(softmax(np.dot(X,theta_sample)),y)
        cost_array[i] = cost
    return theta,cost_array,hocost_array

def train_fit_sgd(alpha,Iter,X,y_text,Xho,yho_text):
        '''
    Implements a stochastic gradient descent algorithm with early stopping based on holdout loss
    Args
        alpha : Learning rate
        Iter : Number of Epochs
        X : Training set
        y : Training labels
        Xho : Holdout set
        yho : Holdout labels
    Returns:
        theta : Best weights
        cost_array : Training cost for each epoch
        hocost_array : Holdout cost for each epoch
        '''

        #One-hot encoding
        y = onehotencoding(y_text)
        yho = onehotencoding(yho_text)

        cost_array = np.zeros(Iter)
        hocost_array = np.zeros(Iter)
        curr_ho_cost = np.inf
        #Adding Intercept --> X becomes n*(d+1) size
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Xho = np.concatenate((np.ones((Xho.shape[0], 1)), Xho), axis=1)
        #creating the parameter vector d+1 parameters
        theta = np.random.rand(X.shape[1],6);
        theta_sample = np.random.rand(X.shape[1],6);
        for i in range(0,Iter):
            cost = 0
            hocost = 0
            #SGD- randomize the order of training set and go through all of
            #them one at a time
            random_order = np.random.permutation(range(0,len(X)))
            for rand_int in random_order:
                X_i = X[rand_int,:]
                y_i = y[rand_int]
                z = np.dot(X_i, theta)
                h = softmax(z)
                gradient = np.dot(np.array([X_i]).T, np.array([h - y_i]))
                theta_sample -= alpha * gradient
            cost += loss(softmax(np.dot(X,theta_sample)),y)
            cost_array[i]  = cost
            hocost += loss(softmax(np.dot(Xho,theta_sample)),yho)
            hocost_array[i] = hocost
            #Early stopping
            if hocost < curr_ho_cost:
                theta = theta_sample;
                curr_ho_cost = hocost;
        return theta,np.array(cost_array),hocost_array

def ypred(X,theta):
    ''' Predicts the class by checking the maximum of w^Tx
    Args
        X : Test set
        theta : Learned weights
    Returns
        class : Number (from 0 to n_class) which predicts the class number th test sample belongs to'''
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return np.argmax(X.dot(theta),axis = 1)


n_epoch = 50; # no of Iterations for training via gradient descent
n_shuffle = 10;# no of iteration for each subject to be a test set
ndims = 40
alphas= np.array([0.05])
colors = ['b','g','r','k','c','m']
colorCounter = 0
test_Acc = []

#Iterate over all alphas
for alpha in alphas:
    costArrays = []
    hoCostArrays = []
    sgdhoCostArrays = []
    sgdcostArrays = []
    testAccuracy = []
    confusionMatrix = np.zeros((6,6))

    #Run for n_shuffle number of iterations
    for i in range(0,n_shuffle):
        original_images,train_images,train_labels,holdout_images,holdout_labels,test_images,test_labels,evalues,evectors = load_train_test_images(i, flag = "all",PCA_dimensions = ndims)
        CoefTheta,costArray,hoCostArray = train_fit_gradient_descent(alpha,n_epoch,train_images,train_labels,holdout_images,holdout_labels)
        costArrays.append(costArray)
        hoCostArrays.append(hoCostArray)
        trainpred = ypred(train_images,CoefTheta)
        testpred = ypred(test_images,CoefTheta)
        test_intencoded = intencode(test_labels)
        #sgdCoefTheta,sgdcostArray,sgdhoCostArray = train_fit_sgd(alpha,n_epoch,train_images,train_labels,holdout_images,holdout_labels)
        #sgdcostArrays.append(sgdcostArray)
        #sgdhoCostArrays.append(sgdhoCostArray)
        #testAccuracy.append(np.average(testpred == test_labels))

        #Compute the confusion matrix
        for i in range(len(test_intencoded)):
            confusionMatrix[test_intencoded[i],testpred[i]] += 1


    costArrays = np.array(costArrays)
    hoCostArrays = np.array(hoCostArrays)
    #sgdcostArrays = np.array(sgdcostArrays)
    #sgdhoCostArrays = np.array(sgdhoCostArrays)

    #Plotting wrappers
    plt.figure(1)
    plt.plot(range(1,n_epoch+1),costArrays.mean(axis = 0),colors[colorCounter],label = str(alpha))
    plt.errorbar(x = np.array([10, 20, 30, 40, 50]),y = costArrays.mean(axis = 0)[[9, 19, 29, 39, 49]],yerr = costArrays.std(axis = 0)[[9, 19, 29, 39, 49]],fmt= colors[colorCounter]+"o")
    plt.legend()
    plt.title(r"\bfseries{Training cost vs Epoch averaged over 10 iterations}")
    plt.xlabel(r"\bfseries{Epoch}")
    plt.ylabel(r"\bfseries{Cost}")
    #plt.savefig(str(ndims)+"_PCA_50 epochs_GD_Train_cost"+str(alpha)+".pdf")
    #plt.savefig(str(ndims)+"_PCA_"+str(n_epoch)+"epochs_GD0.01_Train_cost.pdf")

    plt.figure(2)
    plt.plot(range(1,n_epoch+1),hoCostArrays.mean(axis = 0),colors[colorCounter],label = str(alpha))
    plt.errorbar(x = np.array([10, 20, 30, 40, 50]),y = hoCostArrays.mean(axis = 0)[[9, 19, 29, 39, 49]],yerr = hoCostArrays.std(axis = 0)[[9, 19, 29, 39, 49]],fmt = colors[colorCounter]+"o")
    plt.title(r"\bfseries{Holdout cost vs Epoch GD over 10 iterations}")
    plt.xlabel(r"\bfseries{Epoch}")
    plt.ylabel(r"\bfseries{Cost}")
    plt.legend()
    #plt.savefig(str(ndims)+"_PCA_"+str(n_epoch)+"epochs_GD0.01to0.1Hold_cost.pdf")
    #plt.savefig(str(ndims)+"_PCA_50 epochs_GD_SGD_Err_Train_Holdout_cost"+str(alpha)+".pdf")
    #plt.savefig("40_PCA_50 epochs_GD_Holdout_cost.pdf")
    colorCounter += 1

    confusionMatrix = confusionMatrix / (np.array([np.sum(confusionMatrix,axis = 1)]).T)

    print(confusionMatrix)
    testAccuracy = np.average(confusionMatrix.diagonal())*100
    test_Acc.append(testAccuracy)
    print("test accuracy = ",testAccuracy,"% - "+str(alpha))
    #print("test accuracy = ",testAccuracy.mean())


#Weight visualization
transformed = np.matmul(CoefTheta[1:41,:].T,evectors.T)
for i in range(0,len(transformed)):
    transformed[i] = ((transformed[i] - transformed[i].min()) * (1/(transformed[i].max() - transformed[i].min()) * 255)).astype('uint8')

#plt.figure()

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3)
fig.tight_layout()


#plt.subplot(231)
ax1.imshow(transformed[0].reshape(380,240), cmap = 'gray')
ax1.set_title(r"\bfseries{Happy}")
#plt.subplot(232)
ax2.imshow(transformed[1].reshape(380,240), cmap = 'gray')
ax2.set_title(r"\bfseries{Maudlin}")
#plt.subplot(233)
ax3.imshow(transformed[2].reshape(380,240), cmap = 'gray')
ax3.set_title(r"\bfseries{Surprise}")
#plt.subplot(234)
ax4.imshow(transformed[3].reshape(380,240), cmap = 'gray')
ax4.set_title(r"\bfseries{Fear}")
#plt.subplot(235)
ax5.imshow(transformed[4].reshape(380,240), cmap = 'gray')
ax5.set_title(r"\bfseries{Anger}")
#plt.subplot(236)
ax6.imshow(transformed[5].reshape(380,240), cmap = 'gray')
ax6.set_title(r"\bfseries{Disgust}")
fig.subplots_adjust(wspace=0.2)
plt.show()
#fig.savefig("Weights_"+str(ndims)+"PCA_GD_0.1.pdf", bbox_inches='tight')
print(test_Acc)

