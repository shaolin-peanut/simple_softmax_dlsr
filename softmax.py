import pandas as pd
import numpy as np
import math

def cross_entropy_loss(true_labels, predictions):
    return -np.sum(true_labels * np.log(predictions), axis=0)

def softmax(x_t, y_t):

    # normalize
    x_t_min = x_t.min(axis=0)
    x_t_max = x_t.max(axis=0)
    x_t = (x_t - x_t_min) / (x_t_max - x_t_min)

    featurec = x_t.shape[1]
    classc = y_t.shape[1]
    weights = np.random.randn(classc, featurec)

    netinput = np.matmul(x_t, weights.T)
    # netinput_stable = netinput - np.max(netinput, axis=1, keepdims=True)

    zexp = np.exp(netinput)
    expsum = np.sum(zexp, axis=1, keepdims=True)
    softmax_probabilities = zexp / (expsum + 1e-4)

    print(cross_entropy_loss(y_t, softmax_probabilities))

def main():

    data = pd.read_csv("datasets/dataset_train.csv").dropna()

    rowsc = data.shape[0]
    valrowsc = int(rowsc * 0.2)
    # * 0.2 gets 20% percent of rows count

    classes = data[['Hogwarts House']]
    # one hot encoding
    y = pd.get_dummies(classes, columns = ["Hogwarts House"])

    # traning set to initialize input features, removed all columns containing strings for simplicity, for now
    x_t = data.drop(
        ['Hogwarts House', 
        'Birthday', 
        'Best Hand', 
        'Index', 
        'First Name', 
        'Last Name'], axis=1).values[:-valrowsc]
    y_t = y[:-valrowsc]

    softmax(x_t, y_t)
    # validation set (to run the cost function later)
    x_v = data.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1).values[-valrowsc:] # first 20% of data
    # normalize too
    # x_v = (x_v - x_t_min) / (x_t_max - x_t_min)


if __name__ == "__main__":
    main()