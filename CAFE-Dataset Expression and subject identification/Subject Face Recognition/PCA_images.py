import numpy as np
import matplotlib.pyplot as plt
import dataloader

def find_evalues(train_set,flag = "none",ndims = 6,imshape = (10,10)):
    '''
    Implements PCA from scratch and finds eigenvectors and eigenvalues
    by using a smaller dimension covariance matrix
    Args
        train_set : Matrix to find PCA components for
        flag : Flag to plot PCA components
        ndims : Number of PCA dimensions
        imshape : Shape of the original images for visualizing PCA components
    Returns
       evalue : Eigenvalues of the covariance matrix
       evector : PCA components (aka eigenvectors of the covariance matrix)
       '''
    #PCA
    train_set = train_set
    #Calculate the smaller dimensional matrix
    R = train_set.dot(train_set.T)/(len(train_set) - 1)
    evalue, evector = np.linalg.eig(R)
    #Calculate the original eigenvectors and normalize them
    evnew = train_set.T.dot(evector)/np.sqrt(len(train_set) - 1)
    idx = np.argsort(evalue)[::-1]
    evalue = evalue[idx]
    evnew = evnew[:,idx]

    return evalue[:ndims],(evnew[:,:ndims]/np.sqrt(evalue[:ndims]))

def traintestIndexsplit(images,labels,i):
    '''
    Splits the dataset into train, holdout and test datasets. The ith image is chosen as the test image, and a
    random image from the set is chosen as the holdout image
    Args
        images : dataset to split
        labels : labels of the dataset
        i : Index of the test image (usually the iteration number)
    Returns
        train_dataset : training dataset
        train_labels : training labels
        holdout_dataset : holdout dataset
        holdout_labels : holdout_labels
        test_dataset : test dataset
        test_lables : test_labels
    '''

    rand_int = np.random.randint(0,len(images))
    while i == rand_int:
        rand_int = np.random.randint(0,len(images))
    #return np.delete(images,[i,rand_int], 0),np.delete(labels,[i,rand_int], 0),images[i],labels[i],images[rand_int],labels[rand_int]
    return train_dataset,train_labels,test_dataset,test_labels,holdout_dataset,holdout_labels

#Global array that will be referenced by the classifier
label_list = []

def load_train_test_images(test_index, flag = "all",PCA_dimensions = 4):
    '''
    Image pre-processor. Loads the images, extracts the labels, removes unncessary images, does zero mean and unit SD, does PCA and sends the images to the training function in a usable form
    Args
        test_index : Number of the iteration (needed for train-holdout-test split)
        flag : Used to choose between the three problems - happymaudlin, afraidsurprised, all (softmax)
        PCA_dimensions : Number of PCA dimensions required
    Returns
       train_set : Original images
       PCA_images : Projected images
       train_labels : Training labels
       PCA_holdout : Projected holdout images
       holdout_labels : Holdout labels
       PCA_test : Projected test images
       test_labels : Test labels
       evalues : Eigenvalues
       evectors : PCA components (eigenvectors)
    '''

    images, labels = dataloader.load_data("../../CAFE/CAFE/")
    imshape = images[0].shape
    imagesWONeutrals = []
    labelsWONeutrals = []
    global label_list

    for i in range(len(labels)):
        imagesWONeutrals.append(images[i].flatten())
        labelsWONeutrals.append(labels[i][0:3])
        if labels[i][0:3] not in label_list:
            label_list.append(labels[i][0:3])

    #Remnant of the previous section - we don't discard neutrals or ht
    imagesWONeutrals = np.array(imagesWONeutrals)
    labelsWONeutrals = np.array(labelsWONeutrals)
    label_list = np.array(label_list)


    train_set = []
    train_labels = []
    holdout_set = []
    holdout_labels = []
    test_set = []
    test_labels = []

    #Split images by subject name
    for label in label_list:
        train_images,temp_train_labels,holdout_images,temp_holdout_labels,test_images,temp_test_labels = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == label],labelsWONeutrals[labelsWONeutrals == label],test_index)
        train_set.extend(train_images)
        train_labels.extend(temp_train_labels)
        holdout_set.append(holdout_images)
        holdout_labels.append(temp_holdout_labels)
        test_set.append(test_images)
        test_labels.append(temp_test_labels)


    train_set = np.array(train_set)
    train_labels = np.array(train_labels)
    holdout_set = np.array(holdout_set)
    holdout_labels = np.array(holdout_labels)
    test_set = np.array(test_set)
    test_labels = np.array(test_labels)

    #Zero mean and unit SD before PCA
    train_average = np.average(train_set,axis = 0)
    train_std = np.std(train_set,axis = 0)
    train_set = (train_set - np.average(train_set,axis = 0))/np.std(train_set,axis = 0)
    holdout_set = (holdout_set - train_average)/train_std
    test_set = (test_set - train_average)/train_std


    evalues,evectors = find_evalues(train_set,ndims = PCA_dimensions,imshape = imshape)
    PCA_images = train_set.dot(evectors)
    #Implemented the "sanity check" from Piazza for projecting
    #onto the eigenvectors
    PCA_std = np.sqrt(evalues)
    PCA_images /= PCA_std
    PCA_holdout = holdout_set.dot(evectors)/PCA_std
    PCA_test = test_set.dot(evectors)/PCA_std
    return train_set,PCA_images,train_labels,PCA_holdout,holdout_labels,PCA_test,test_labels,evalues,evectors

