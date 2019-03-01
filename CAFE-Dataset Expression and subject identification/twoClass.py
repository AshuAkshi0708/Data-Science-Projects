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

def sigmoid(z):
    '''Sigmoid function
    Args
        z : w^Tx
    Returns
        sigmoid(z)'''
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    '''
    Implements Logistic regression loss
    Args
        h : w^Tx
        y : Train labels
    Returns
        Logistic regression Loss
    '''
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def train_fit_gradient_descent(alpha,Iter,X,y,Xho,yho):
        '''
    Implements a batch gradient descent algorithm with early stopping based on holdout loss
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
        cost_array = np.zeros(Iter);
        hocost_array = np.zeros(Iter);
        curr_ho_cost = np.inf
        #Adding Intercept --> X becomes n*(d+1) size
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Xho = np.concatenate((np.ones((Xho.shape[0], 1)), Xho), axis=1)
        #creating the parameter vector d+1 parameters
        theta = np.random.random(X.shape[1]);
        theta_sample = np.random.random(X.shape[1]);
        #Start gradient descent
        for i in range(0,Iter):
            cost = 0;
            hocost = 0;
            z = np.dot(X, theta)
            h = sigmoid(z)
            #Compute gradient
            gradient = np.dot(X.T, (h - y))
            theta_sample -= alpha * gradient
            hocost += loss(sigmoid(np.dot(Xho,theta_sample)),yho)
            hocost_array[i] = hocost
            #Early stopping
            if (hocost) < curr_ho_cost:
                theta = theta_sample;
                curr_ho_cost = hocost;
            cost+= loss(sigmoid(np.dot(X,theta_sample)),y)
            cost_array[i] = cost
        return theta,cost_array,hocost_array


def pred_prob(X,theta):
    '''
    Computes the probability for test prediction
    Args
        X - Test sample
        theta - weights
    Returns
        prob : Sigmoid probability
    '''
    #Add one in front for the bias
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return sigmoid(np.dot(X,theta))

def ypred(y):
    '''
    Predicts the class based on test probability
    Args
        y - sigmoid probability
    Returns
        class - 0 or 1
    '''
    return np.around(y)



n_epoch = 10 # no of Iterations for training via gradient descent
n_shuffle = 10# no of iteration for each subject to be a test set
#alphas= np.array([0.0001])
#alphas = np.array([1e-4,0.01,0.1])
#alphas = np.array([1e-4,1e-2,0.1])
alphas = np.array([0.01,0.1,1])

colors = ['b','g','r','k','c','m'] #Plotting stuff
colorCounter = 0 #Plotting stuff

#Iterate over all alphas
for alpha in alphas:
    costArrays = []
    hoCostArrays = []
    testAccuracy = []
    #Run for n_shuffle number of iterations
    for i in range(0,n_shuffle):
        original_images,train_images,train_labels,holdout_images,holdout_labels,test_images,test_labels,evalues,evectors = load_train_test_images(i, flag="happymaudlin",PCA_dimensions = 4)
        CoefTheta,costArray,hoCostArray = train_fit_gradient_descent(alpha,n_epoch,train_images,train_labels,holdout_images,holdout_labels)
        costArrays.append(costArray)
        hoCostArrays.append(hoCostArray)
        trainpred = ypred(pred_prob(train_images,CoefTheta))
        testpred = ypred(pred_prob(test_images,CoefTheta))
        testAccuracy.append(np.average(testpred == test_labels))

    #Append all cost arrays from all the iterations
    costArrays = np.array(costArrays)
    hoCostArrays = np.array(hoCostArrays)

    #Plotting wrappers
    plt.figure(50)
    plt.plot(range(1,n_epoch+1),costArrays.mean(axis = 0),colors[colorCounter],label = str(alpha))
    plt.errorbar(x = np.array([2,4,8,10]),y = costArrays.mean(axis = 0)[[1,3,7,9]],yerr = costArrays.std(axis = 0)[[1,3,7,9]],fmt= colors[colorCounter]+"o")
    plt.legend()
    plt.title(r"\bfseries{Training error vs Epoch \\averaged over 10 iterations}")
    plt.xlabel(r"\bfseries{Epoch}")
    plt.ylabel(r"\bfseries{Training cost}")
    plt.figure(51)
    plt.plot(range(1,n_epoch+1),hoCostArrays.mean(axis = 0),colors[colorCounter],label = str(alpha))
    plt.errorbar(x = np.array([2,4,8,10]),y = hoCostArrays.mean(axis = 0)[[1,3,7,9]],yerr = hoCostArrays.std(axis = 0)[[1,3,7,9]],fmt = colors[colorCounter]+"o")
    plt.title(r"\bfseries{Holdout cost function vs Epoch averaged over 10 iterations}")
    plt.xlabel(r"\bfseries{Epoch}")
    plt.ylabel(r"\bfseries{Holdout cost}")
    plt.legend()

    plt.figure()
    plt.plot(range(1,n_epoch+1),costArrays.mean(axis = 0),colors[colorCounter],label = str(alpha)+"-Train")
    plt.errorbar(x = np.array([2,4,8,10]),y = costArrays.mean(axis = 0)[[1,3,7,9]],yerr = costArrays.std(axis = 0)[[1,3,7,9]],fmt= colors[colorCounter]+"o")
    plt.plot(range(1,n_epoch+1),hoCostArrays.mean(axis = 0),colors[colorCounter+1],label = str(alpha)+"-Holdout")
    plt.errorbar(x = np.array([2,4,8,10]),y = hoCostArrays.mean(axis = 0)[[1,3,7,9]],yerr = hoCostArrays.std(axis = 0)[[1,3,7,9]],fmt = colors[colorCounter+1]+"o")
    plt.legend()

    plt.title(r"Errors for the best classifier")
    plt.xlabel(r"\bfseries{Epoch}")
    plt.ylabel(r"\bfseries{Errors}")

    testAccuracy = np.array(testAccuracy)
    print("test accuracy = {} \pm {}".format(testAccuracy.mean(),testAccuracy.std()))
    colorCounter += 1
plt.show()


