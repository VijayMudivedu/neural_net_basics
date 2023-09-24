import numpy as np
import sklearn.datasets
import sklearn.linear_model
import  matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Load dataset

def load_dataset():
    np.random.seed(2)
    # No. of samples
    m = 400 # no. of samples
    D = 2   # dimensionality in X
    N = int(m/2) # samples per class
    X = np.zeros(shape=(m,D))
    y = np.zeros(shape=(m,1), dtype='uint8')
    a = 4 # no. radii in the flow

    # maximum rays in the flower 4
    for j in range(2):
        ix = range(N*j, N*(j+1))
        theta = np.linspace(j*np.pi,(j+1)*np.pi,N) + np.random.randn(N) * 0.2  # theta
        radius = a*np.sin(4*theta) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[ix] = j
    X = X.T
    y = y.T

    return X,y


# plot decision boundary
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure