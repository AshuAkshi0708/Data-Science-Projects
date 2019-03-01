import numpy as np
import matplotlib.pyplot as plt
from PCA_images import load_train_test_images,label_list

def softmax(z):
    '''Implements the softmax function for a vector containing w^Tx for the weights corresponding to each class
    Args
        z : Vector containing w^Tx
    Returns
        sm : Softmax vector containing the softmax probabilities for all the classes'''

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

    encode = dict((l, i) for i, l in enumerate(label_list))
    decode = dict((i, l) for i, l in enumerate(label_list))
    # integer encode input data
    int_encoded = [encode[l] for l in labels]
    #print(int_encoded)
    # one hot encode
    onehotCoded = list()
    for value in int_encoded:
        letter = [0 for i in range(len(label_list))]
        letter[value] = 1
        onehotCoded.append(letter)
    # invert encoding
    inverseCoded = decode[np.argmax(onehotCoded[0])]
    #print(inverseCoded)
    return np.array(onehotCoded)

def intencode(labels):
    encode = dict((l, i) for i, l in enumerate(label_list))
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

        y = onehotencoding(y_text)
        yho = onehotencoding(yho_text)
        cost_array = np.zeros(Iter);
        hocost_array = np.zeros(Iter);
        curr_ho_cost = np.inf
        #Adding Intercept --> X becomes n*(d+1) size
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Xho = np.concatenate((np.ones((Xho.shape[0], 1)), Xho), axis=1)
        #creating the parameter vector d+1 parameters
        theta = np.zeros((X.shape[1],10)) #HARDCODE!!!!!
        theta_sample = np.zeros((X.shape[1],10))
        for i in range(0,Iter):
            cost = 0;
            hocost = 0;
            z = np.dot(X, theta)
            h = softmax(z)
            gradient = np.dot(X.T, (h - y))
            theta_sample -= alpha * gradient
            hocost += loss(softmax(np.dot(Xho,theta_sample)),yho)
            hocost_array[i] = hocost
            if (hocost) < curr_ho_cost:
                theta = theta_sample;
                curr_ho_cost = hocost;
            cost+= loss(softmax(np.dot(X,theta_sample)),y)
            cost_array[i] = cost
        return theta,cost_array,hocost_array

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
n_shuffle = 8;# no of iteration for each subject to be a test set
#alphas= np.array([0.001,0.01,0.1])
alphas = np.array([0.001])

#Iterate over all alphas
for alpha in alphas:
    costArrays = []
    hoCostArrays = []
    testAccuracy = []
    confusionMatrix = np.zeros((10,10))

    #Run for n_shuffle number of iterations
    for i in range(0,n_shuffle):


        original_images,train_images,train_labels,holdout_images,holdout_labels,test_images,test_labels,evalues,evectors = load_train_test_images(i, flag = "all",PCA_dimensions = 10)
        CoefTheta,costArray,hoCostArray = train_fit_gradient_descent(alpha,n_epoch,train_images,train_labels,holdout_images,holdout_labels)
        costArrays.append(costArray)
        hoCostArrays.append(hoCostArray)
        trainpred = ypred(train_images,CoefTheta)
        testpred = ypred(test_images,CoefTheta)
        test_intencoded = intencode(test_labels)

        #Compute the confusion matrix
        for i in range(len(test_intencoded)):

            confusionMatrix[test_intencoded[i],testpred[i]] += 1


    costArrays = np.array(costArrays)
    hoCostArrays = np.array(hoCostArrays)
    plt.figure(1)
    plt.plot(costArrays.mean(axis = 0),label = str(alpha))
    plt.title("Cost array")
    plt.legend()
    plt.figure(2)
    plt.plot(hoCostArrays.mean(axis = 0),label = str(alpha))
    plt.title("hoCost array")
    plt.legend()
    testAccuracy = np.array(testAccuracy)

    confusionMatrix = confusionMatrix / (np.array([np.sum(confusionMatrix,axis = 1)]).T)

    print(confusionMatrix)

    #Weight visualization
    weights = CoefTheta[1:,].T.dot(evectors.T)
    plt.figure()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(weights[i].reshape(380,240),cmap = "gray")
        plt.title(label_list[i])
        plt.axis('off')


plt.show()


