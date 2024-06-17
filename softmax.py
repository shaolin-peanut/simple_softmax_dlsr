import pandas as pd
import numpy as np
import math

epochs = 300
learning_rate = 0.01


def setup_data():
    data = pd.read_csv("datasets/dataset_train.csv").dropna()

    # one hot encoding of classes (houses)
    y = pd.get_dummies(data[['Hogwarts House']], columns = ["Hogwarts House"])

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

def cross_entropy_loss(true_labels, predictions):
    return -np.sum(true_labels * np.log(predictions + 1e-9), axis=1).mean()

def gradients(pred, true_values, inputlayer):
    diff = pred - true_values
    weight_gradient = np.matmul(diff.T, inputlayer)
    return weight_gradient

def update_weights(gradient, weight):
    return (weight - (learning_rate * gradient))

def softmax(inputlayer):
    zexp = np.exp(inputlayer - np.max(inputlayer, axis=1, keepdims=True)).astype(np.float128)
    expsum = np.sum(zexp, axis=1, keepdims=True)
    return zexp / expsum

def training(x_t, y_t, weight):
    for epoch in range(epochs):
        inputlayer = np.matmul(x_t, weight.T)
        inputlayer_s = inputlayer - np.max(inputlayer, axis=1, keepdims=True)

        softmax_probabilities = softmax(inputlayer_s)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Probabilities Sum: {np.sum(softmax_probabilities, axis=1)}")

        loss = np.average(cross_entropy_loss(y_t, softmax_probabilities))
        print(f"Epoch {epoch + 1}, Loss: {loss}")

        gradient_vector = gradients(softmax_probabilities, y_t, x_t)

        weight = update_weights(gradient_vector, weight)
    print(softmax_probabilities)
    return weight

def main():
    x_t, y_t = setup_data()

    feature_count = x_t.shape[1]
    class_count = y_t.shape[1]
    initial_weights = np.random.randn(class_count, feature_count) * 0.01

    weights = training(x_t, y_t, initial_weights)

if __name__ == "__main__":
    main()