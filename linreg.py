import numpy as np



def load_data(path, num_train):
    #loading data into an np array
    data = np.loadtxt(path,delimiter=';',skiprows=1,unpack=False)
    #Segrgating the array into different datasets
    X_train = np.array(data[:num_train,:11])
    Y_train = np.array(data[:num_train,11])
    X_test = np.array(data[num_train:,:11])
    Y_test = np.array(data[num_train:,11])
    
    """ Load the data matrices
    Input:
    path: string describing the path to a .csv file
          containing the dataset
    num_train: number of training samples
    Output:
    X_train: numpy array of shape num_train x 11
             containing the first num_train many
             data rows of columns 1 to 11 of the
             .csv file.
    Y_train: numpy array of shape num_train
             containing the first num_train many
             data rows of column 12 of the .csv
             file.
    X_test: same as X_train only corresponding to
            the remaining rows after the first 
            num_train many rows.
    Y_test: same as Y_train only corresponding to
            the remaining rows after the first 
            num_train many rows.
    """
    # TODO: load data according to the specifications,
    # e.g. using numpy.loadtxt
    
    return X_train, Y_train, X_test, Y_test


def fit(X, Y):
    #Append an additional colunm due to the bias theta12
    x = np.append(X,np.ones((len(X),1)), axis = 1)
    #Calculation of theta is carried out partwise
    theta1 = np.linalg.inv(np.matmul(x.T,x))
    theta2 = np.matmul(x.T,Y)
    theta = np.matmul(theta1,theta2)
    
    """ Fit linear regression model
    Input:
    X: numpy array of shape N x n containing data
    Y: numpy array of shape N containing targets
    Output:
    theta: nump array of shape n + 1 containing weights
           obtained by fitting data X to targets Y
           using linear regression
    """
    # TODO
    return theta


def predict(X, theta):
    x= np.append(X,np.ones((len(X),1)), axis = 1)
    Y_pred = np.matmul(x,theta)
    
    """ Perform inference using data X
        and weights theta
    Input:
    X: numpy array of shape N x n containing data
    theta: numpy array of shape n + 1 containing weights
    Output:
    Y_pred: numpy array of shape N containig predictions
    """
    # TODO
    return Y_pred


def energy(Y_pred, Y_gt):
    se = np.sum((Y_pred-Y_gt)**2)
    """ Calculate squared error
    Input:
    Y_pred: numpy array of shape N containing prediction
    Y_gt: numpy array of shape N containing targets
    Output:
    se: squared error between Y_pred and Y_gt
    """
    # TODO
    return se
