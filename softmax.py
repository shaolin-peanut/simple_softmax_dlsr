import pandas as pd
import numpy as np
import math

epochs = 2000
learning_rate = 0.01

def setup_data():
    data = pd.read_csv("datasets/dataset_train.csv").dropna()

    # one hot encoding of classes (houses)
    y = pd.get_dummies(data[['Hogwarts House']], dtype=int)

    # traning set to initialize input features, removed all columns containing strings for simplicity, for now
    x_t = data.drop(
        ['Hogwarts House', 
        'Birthday', 
        'Best Hand',
        'Index',
        'First Name',
        'Last Name'], axis=1).values
    y_t = y.values

    # normalize
    x_t_min = x_t.min(axis=0)
    x_t_max = x_t.max(axis=0)
    x_t = (x_t - x_t_min) / (x_t_max - x_t_min)
    
    return x_t, y_t

# def cross_entropy_loss(true_labels, predictions):
#     return -np.sum(true_labels * np.log(predictions + 1e-9), axis=1).mean()

# def gradients(pred, true_values, inputlayer):
#     diff = pred - true_values
#     weight_gradient = np.matmul(diff.T, inputlayer)
#     return weight_gradient

#     log = log_softmax()


# def jacobian_matrix(inputlayer):
#     softmaxlog = log_softmax(inputlayer)
    # ∂log(softmax(z)i​)​=δik​−∑j=1n​ezj​ezk​​

import sklearn.metrics as sk    

def calculate_accuracy(truth, pred):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(truth, axis=1)
    return np.mean(pred_labels == true_labels)

def calculate_precision(truth, pred):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(truth, axis=1)
    return sk.precision_score(true_labels, pred_labels, average='macro')

def calculate_recall(truth, pred):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(truth, axis=1)
    return sk.recall_score(true_labels, pred_labels, average='macro')

def calculate_f1(truth, pred):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(truth, axis=1)
    return sk.f1_score(true_labels, pred_labels, average='macro')

def softmax(inputlayer):
    zexp = np.exp(inputlayer - np.max(inputlayer, axis=1, keepdims=True))
    expsum = np.sum(zexp, axis=1, keepdims=True)
    return zexp / expsum

def categorical_cross_entropy(truth, pred, inputl):
    # simplified version to reduce computation steps
    result = -np.sum(truth * np.log(pred)) / truth.shape[0]
    return result

    # the 'full' function that is typically done, more expensive
    # s = softmax(inputl)
    # log = np.log(s)
    # return (- (truth * log)

def print_metrics(epoch, y_t, s):
    accuracy = calculate_accuracy(y_t, s)
    precision = calculate_precision(y_t, s)
    recall = calculate_recall(y_t, s)
    f1 = calculate_f1(y_t, s)
    # print(f"Epoch {epoch + 1}, Loss: {loss}")
    print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")
    print(f"Epoch {epoch + 1}, Precision: {precision}")
    print(f"Epoch {epoch + 1}, Recall: {recall}")
    print(f"Epoch {epoch + 1}, F1 Score: {f1}")

def training(x_t, y_t, weight):
    for epoch in range(epochs):
        inputlayer = np.matmul(x_t, weight.T)
        # inputlayer_s = inputlayer - np.max(inputlayer, axis=1, keepdims=True)

        s = softmax(inputlayer)
        # loss = categorical_cross_entropy(y_t, s, inputlayer)

        if epoch % 100 == 0: print_metrics(epoch, y_t, s)

        # compute the gradient
        gradient_vector = (s - y_t)  / x_t.shape[0]

        weight -= learning_rate * np.matmul(gradient_vector.T, x_t)
    return weight

def main():
    x_t, y_t = setup_data()

    feature_count = x_t.shape[1]
    class_count = y_t.shape[1]
    initial_weights = np.random.randn(class_count, feature_count) * 0.01

    weights = training(x_t, y_t, initial_weights)

if __name__ == "__main__":
    main()
