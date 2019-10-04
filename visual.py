import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def setup(test_size = 0.3, random_state = 0):
    """Setup data for training."""
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size = test_size, random_state = random_state)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
    """A convenient function for plotting decision regions"""

    # Setup markers generator and color map.
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all training samples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c =
                cmap(idx), marker = markers[idx], label = cl)

    # Highlight test samples.
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(x = X_test[:, 0], y = X_test[:, 1], c = "",
                alpha = 1.0, linewidth = 1, marker = "o", s = 55,
                edgecolor = "black", label = "test set")

def plot_cost_function(eta1, n_iter1, eta2, n_iter2, X, y):
    """Plot the cost function against the number of epochs
    for the two different learning rates"""

    # Import Adaline class.
    from main import AdalineGD

    # Setup a subplot for comparison.
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))

    # Setup the first Adaline.
    ada1 = AdalineGD(eta1, n_iter2).fit(X, y)
    print(ada1.cost_)
    ax[0].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker = "o")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Sum-squared-error")
    ax[0].set_title("Adaline - learning rate " + str(eta1))

    # Setup the second Adaline.
    ada2 = AdalineGD(eta2, n_iter2).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = "o")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Sum-squared-error")
    ax[1].set_title("Adaline - learning rate " + str(eta2))







