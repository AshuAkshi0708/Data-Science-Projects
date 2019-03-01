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
    #Calculate the other smaller dimensional matrix
    R = train_set.dot(train_set.T)/(len(train_set) - 1)
    evalue, evector = np.linalg.eig(R)
    #Calculate the original eigenvectors and normalize them
    evnew = train_set.T.dot(evector)/np.sqrt(len(train_set) - 1)
    idx = np.argsort(evalue)[::-1]
    evalue = evalue[idx]
    evnew = evnew[:,idx]
    print("ndims",ndims)
    if flag == "plotPCA":
        plt.figure()
        plt.subplot(231)
        plt.imshow(evalue[0] * evnew[:,0].reshape(imshape),cmap = "gray")
        plt.subplot(232)
        plt.imshow(evalue[1] * evnew[:,1].reshape(imshape),cmap = "gray")
        plt.subplot(233)
        plt.imshow(evalue[2] * evnew[:,2].reshape(imshape),cmap = "gray")
        plt.subplot(234)
        plt.imshow(evalue[3] * evnew[:,3].reshape(imshape),cmap = "gray")
        plt.subplot(235)
        plt.imshow(evalue[4] * evnew[:,4].reshape(imshape),cmap = "gray")
        plt.subplot(236)
        plt.imshow(evalue[5] * evnew[:,5].reshape(imshape),cmap = "gray")

        plt.show()

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
    train_dataset = np.delete(images,[i,rand_int], 0)
    train_labels = np.delete(labels,[i,rand_int],0)
    test_dataset = images[i]
    test_labels = labels[i]
    holdout_dataset = images[rand_int]
    holdout_labels = labels[rand_int]
    #return np.delete(images,[i,rand_int], 0),np.delete(labels,[i,rand_int], 0),images[i],labels[i],images[rand_int],labels[rand_int]
    return train_dataset,train_labels,test_dataset,test_labels,holdout_dataset,holdout_labels


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
    images, labels = dataloader.load_data("../CAFE/CAFE/")
    imshape = images[0].shape
    imagesWONeutrals = []
    labelsWONeutrals = []

    emotion_dict = {"h": "happy", "ht": "happy with teeth", "m": "maudlin","s": "surprise", "f": "fear", "a": "anger", "d": "disgust", "n": "neutral"}

    for i in range(len(labels)):
        #Remove n and ht from dataset
        if labels[i][-6] == "n" or "ht" in labels[i]:
            continue
        imagesWONeutrals.append(images[i].flatten())
        labelsWONeutrals.append(emotion_dict[labels[i][-6]])

    imagesWONeutrals = np.array(imagesWONeutrals)
    labelsWONeutrals = np.array(labelsWONeutrals)

    #Split the loaded images into six sets depending on the emotions
    happy_train_images,happy_train_labels,happy_test_images,happy_test_labels,happy_hold_images, happy_hold_labels  = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == "happy"], labelsWONeutrals[labelsWONeutrals == "happy"],test_index)
    maudlin_train_images,maudlin_train_labels,maudlin_test_images,maudlin_test_labels,maudlin_hold_images, maudlin_hold_labels = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == "maudlin"],labelsWONeutrals[labelsWONeutrals == "maudlin"],test_index)
    surprise_train_images,surprise_train_labels,surprise_test_images,surprise_test_labels,surprise_hold_images, surprise_hold_labels = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == "surprise"],labelsWONeutrals[labelsWONeutrals == "surprise"],test_index)
    fear_train_images,fear_train_labels,fear_test_images,fear_test_labels,fear_hold_images, fear_hold_labels = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == "fear"],labelsWONeutrals[labelsWONeutrals == "fear"],test_index)
    anger_train_images,anger_train_labels,anger_test_images,anger_test_labels,anger_hold_images, anger_hold_labels = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == "anger"],labelsWONeutrals[labelsWONeutrals == "anger"],test_index)
    disgust_train_images,disgust_train_labels,disgust_test_images,disgust_test_labels,disgust_hold_images, disgust_hold_labels = traintestIndexsplit(imagesWONeutrals[labelsWONeutrals == "disgust"],labelsWONeutrals[labelsWONeutrals == "disgust"],test_index)

    #Plot individual images when required
    #plt.figure()
    #plt.subplot(231)
    #plt.imshow(happy_train_images[0].reshape(images[0].shape),cmap = "gray")
    #plt.title("Happy")
    #plt.subplot(232)
    #plt.imshow(maudlin_train_images[0].reshape(images[0].shape),cmap = "gray")
    #plt.title("Maudlin")
    #plt.subplot(233)
    #plt.imshow(surprise_train_images[0].reshape(images[0].shape),cmap = "gray")
    #plt.title("Surprise")
    #plt.subplot(234)
    #plt.imshow(fear_train_images[0].reshape(images[0].shape),cmap = "gray")
    #plt.title("Fear")
    #plt.subplot(235)
    #plt.imshow(anger_train_images[0].reshape(images[0].shape),cmap = "gray")
    #plt.title("Anger")
    #plt.subplot(236)
    #plt.imshow(disgust_train_images[0].reshape(images[0].shape),cmap = "gray")
    #plt.title("Disgust")

    #Combine the six sets and split them into train, test and holdout
    #depending on the problem
    if flag == "all":
        train_set = np.concatenate((happy_train_images,maudlin_train_images,surprise_train_images,fear_train_images,anger_train_images,disgust_train_images),axis = 0)
        train_labels = np.concatenate((happy_train_labels,maudlin_train_labels,surprise_train_labels,fear_train_labels,anger_train_labels,disgust_train_labels),axis = 0)
        holdout_set = np.concatenate(([happy_hold_images],[maudlin_hold_images],[surprise_hold_images],[fear_hold_images],[anger_hold_images],[disgust_hold_images]),axis = 0)
        holdout_labels = np.concatenate(([happy_hold_labels],[maudlin_hold_labels,surprise_hold_labels],[fear_hold_labels],[anger_hold_labels],[disgust_hold_labels]),axis = 0)
        test_set = np.concatenate(([happy_test_images],[maudlin_test_images],[surprise_test_images],[fear_test_images],[anger_test_images],[disgust_test_images]),axis = 0)
        test_labels = np.concatenate(([happy_test_labels],[maudlin_test_labels],[surprise_test_labels],[fear_test_labels],[anger_test_labels],[disgust_test_labels]),axis = 0)


    elif flag == "happymaudlin":
        train_set = np.concatenate((happy_train_images,maudlin_train_images),axis = 0)
        train_labels = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
        holdout_set = np.concatenate(([happy_hold_images],[maudlin_hold_images]),axis = 0)
        holdout_labels = np.array([1,0])
        test_set = np.concatenate(([happy_test_images],[maudlin_test_images]),axis = 0)
        test_labels = np.array([1,0])

    elif flag == "afraidsurprised":
        train_set = np.concatenate((fear_train_images,surprise_train_images),axis = 0)
        train_labels = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
        holdout_set = np.concatenate(([fear_hold_images],[surprise_hold_images]),axis = 0)
        holdout_labels = np.array([1,0])
        test_set = np.concatenate(([fear_test_images],[surprise_test_images]),axis = 0)
        test_labels = np.array([1,0])

    #Zero mean and Unit SD before PCA
    train_average = np.average(train_set,axis = 0)
    train_std = np.std(train_set,axis = 0)
    train_set = (train_set - np.average(train_set,axis = 0))/np.std(train_set,axis = 0)
    holdout_set = (holdout_set - train_average)/train_std
    test_set = (test_set - train_average)/train_std


    evalues,evectors = find_evalues(train_set,ndims = PCA_dimensions,imshape = imshape,flag = "None")

    #Implemented the "sanity check" from Piazza for projecting
    #onto the eigenvectors
    PCA_images = train_set.dot(evectors)
    PCA_std = np.sqrt(evalues)
    PCA_images /= PCA_std
    PCA_holdout = holdout_set.dot(evectors)/PCA_std
    PCA_test = test_set.dot(evectors)/PCA_std

    return train_set,PCA_images,train_labels,PCA_holdout,holdout_labels,PCA_test,test_labels,evalues,evectors


