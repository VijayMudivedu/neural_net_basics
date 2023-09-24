# Package imports
import os
import matplotlib.pyplot as plt
from neural_net_basics import utils as u
from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(2)

X, Y = u.load_dataset()


def model_linear(X,Y):
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.title("non-linear dataset plot")
    plt.show()

    model = LogisticRegression()
    model.fit(X.T, np.squeeze(Y.T, axis=1))

    lr_pred = model.predict(X=X.T)  # proba(X=X.T)[:,1]
    error = -np.sum(np.dot(Y, np.log(lr_pred + 0.0001)) + np.dot((1 - Y), np.log(1 - lr_pred + 0.0001))) / (
            100 * float(Y.size))
    print(error * 100)
    u.plot_decision_boundary(lambda x: model.predict(x), X, Y)
    plt.title("Logistic Regression")
    plt.show()

# run the linear model
model_linear(X,Y)

# Layer sizes


def layer_sizes(X, Y, n_h=4):
    n_x, n_h, n_y = X.shape[0], n_h, Y.shape[0]
    print(f"shape of nn: INPUT_LAYER:{n_x}, HIDDEN_LAYER: {n_h}, OUTPUT_LAYER: {n_y}")
    return n_x, n_h, n_y


def initialize_params(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.1
    b2 = np.zeros((n_y, 1))
    print(f"shape of init params - W1, b1, W2, b2: {[k.shape for k in [W1, b1, W2, b2]]}")
    return W1, b1, W2, b2


def forward(X, params):
    W1, b1, W2, b2 = params
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = u.sigmoid(Z2)
    y_hat = A2
    cache = (Z1, A1, Z2, A2)
    # print(f"shape of cache: {[k.shape for k in cache]}")
    return y_hat, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    # print(f"A2.shape and Y.shape: {A2.shape, Y.shape}")
    log_probs = np.dot(np.log(A2), Y.T) + np.dot(np.log(1 - A2), (1 - Y).T)
    cost = -(1.0 / m) * np.sum(log_probs)
    cost = np.squeeze(cost)
    cost = float(cost)
    return cost


def backward(params, cache, X, Y):
    m = X.shape[1]
    # initialize params
    W1, b1, W2, b2 = params
    # output computed
    A2 = cache[0]
    # Intermediate value computed
    _, A1, _, A2 = cache[1]
    # gradients
    dZ2 = (A2 - Y)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A2, 2))  # derivative of np.tanh(z) = 1- Z**2
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = (dW1, db1, dW2, db2)
    return grads  # return gradients


def update_params(params, grads, learning_rate=1.2):
    # get the initialized params
    W1, b1, W2, b2 = params
    # get the gradients
    dW1, db1, dW2, db2 = grads
    # update the params
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    updated_parameters = W1, b1, W2, b2
    return updated_parameters  # return updated params


# build the complete model
def nn_model(X, Y, learning_rate=1.2, n_h=4, num_iterations=10000, print_cost=False):
    np.random.seed(5)
    # define the model structure
    n_x, n_h, n_y = layer_sizes(X, Y, n_h)
    # initialize the random number into params
    params = initialize_params(n_x, n_h, n_y)
    # Run the Epochs and iterations and compute the forward
    for i in range(num_iterations):
        # execute the forward pass for each iteration
        A2, cache = forward(X, params=params)
        # compute the cost function
        cost = compute_cost(A2, Y)
        # compute the backward pass
        grads = backward(params, cache, X, Y)
        # feed the gradients to update parameters
        updated_params = update_params(params, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print(f"cost after iteration {i:4d} {cost:.4f}")

    return updated_params


# use this predict the graded function
def predict(params, X, threshold):
    A2, cache = forward(params=params, X=X)
    predicted = A2 > threshold
    return predicted


# Run NN model
final_params = nn_model(X=X,Y=Y, learning_rate=1e-5, n_h=4,print_cost=True)

u.plot_decision_boundary(model=lambda x: predict(params=final_params,X=x.T,threshold=0.5), X=X, y=Y)
plt.title("Decision Boundary")
plt.show()