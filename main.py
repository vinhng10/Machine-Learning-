import random
import numpy as np
from numpy.random import seed

#Perceptron:
class Perceptron():
    """Perceptron classifier."""

    def __init__(self, eta = 0.01, n_iter = 10):
        """
        ---------------
        Parameters:
        eta : float
            Learning rate (between 0.0 and 1.0).
        n_iter : int
            Passes over the training dataset.

        ---------------
        Attribute:
        w_ : 1d-array
            Weights after fitting.
        errors_ : list
            Number of misclassifications in every epoch.

        ---------------
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        ---------------
        Parameters:
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : {array-like}, shape = [n_sample]
            Target values.
        ---------------
        Return:
        self : object.
        ---------------
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit test."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#Gradient Descent:
class AdalineBGD(object):
    """ADAptive LInear NEuron Classifier (Batch Gradient Descent)."""

    def __init__(self, eta = 0.01, n_iter = 50):
        """
        ---------------
        Parameters:
        eta: float
            Learning rate (between 0.0 and 1.0).
        n_iter: int
            Passes over the training dataset.
        ---------------
        Attributes:
        w_: 1d-array
            Weights after fitting.
        errors_: list
            Number of misclassification in each epoch.
        ---------------
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.
        ---------------
        Parameters:
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : {array-like}, shape = [n_sample]
            Target values.
        ---------------
        Return:
        self : object.
        ---------------
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            error = y - output
            self.w_[1:] += self.eta * X.T.dot(error)
            self.w_[0] += self.eta * error.sum()
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after quantizing."""
        return np.where(self.activation(X) >= 0.0, 1, -1)

class AdalineSGD(object):
    """ADAptive LInear NEuron Classifier (Stochastic Gradient Descent)."""

    def __init__(self, eta = 0.01, n_iter = 50, shuffle = True,
            random_state = None):
        """
        ---------------
        Parameters:
        eta: float
            Learning rate (between 0.0 and 1.0).
        n_iter: int
            Passes over the training dataset.
        shuffle: bool (default: True)
            Shuffles the training set
            if True to avoid cycles.
        random_state: int (default: None)
            Set random state for shuffling
            and initializing weights.
        ---------------
        Attributes:
        w_: 1d-array
            Weights after fitting.
        errors_: list
            Number of misclassification in each epoch.
        ---------------
        """
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit training data.
        ---------------
        Parameters:
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : {array-like}, shape = [n_sample]
            Target values.
        ---------------
        Return:
        self : object.
        ---------------
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(cost)
            self.cost_.append(avg_cost)
        return self

    def _partial_fit(self, X, y):
        """Fit training data for Online Learning
        (without initializing without initializing the weights)."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _initialize_weights(self, m):
        """Initialize weights to zeros."""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights."""
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta*error*xi
        self.w_[0] += self.eta*error
        cost = 0.5*(error**2)
        return cost

    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after quantizing."""
        return np.where(self.activation(X) >= 0.0, 1, -1)

class AdalineMBGD(object):
    """ADAptive LInear NEuron Classifier (Mini-Batch Gradient Descent)."""

    def __init__(self, eta = 0.01, n_iter = 50, shuffle = True,
            random_state = None, batch_size = 20):
        """
        ---------------
        Parameters:
        eta: float
            Learning rate (between 0.0 and 1.0).
        n_iter: int
            Passes over the training dataset.
        shuffle: bool (default: True)
            Shuffles the training set
            if True to avoid cycles.
        random_state: int (default: None)
            Set random state for shuffling.
        batch_size: int
            Choose the number of examples in
            mini batch.
        ---------------
        Attributes:
        w_: 1d-array
            Weights after fitting.
        errors_: list
            Number of misclassification in each epoch.
        ---------------
        """
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.batch_size = batch_size
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit training data.
        ---------------
        Parameters:
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : {array-like}, shape = [n_sample]
            Target values.
        ---------------
        Return:
        self : object.
        ---------------
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for num in range(0, len(y), self.batch_size):
                tem_feature = X[num: num + self.batch_size, :]
                tem_target = y[num: num + self.batch_size]
                cost.append(self._update_weights(tem_feature, tem_target))
            avg_cost = sum(cost)/len(cost)
            self.cost_.append(avg_cost)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, X, y):
        """Apply Adaline learning rule to update the weights."""
        output = self.net_input(X)
        error = y - output
        self.w_[1:] += self.eta * X.T.dot(error)
        self.w_[0] += self.eta * error.sum()
        cost = (error**2).sum() / 2.0
        return cost

    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after quantizing."""
        return np.where(self.activation(X) >= 0.0, 1, -1)


#Decision Tree:
class Node(object):
    """Class holding information about a Node in a decision tree."""

    def __init__(self, data = None, question = None, left = None, right =
            None, depth = None, label = None, kind = "Node"):
        """Store information of the node."""
        self.data = data
        self.question = question
        self.left = left
        self.right = right
        self.depth = depth
        self.label = label
        self.kind = kind

class DecisionTree(object):
    """Decision Tree Classifier."""

    def __init__(self, max_depth = 10, min_samples_split = 2, min_samples_leaf
            = 1):
        """Initialize the tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = Node(depth = 0, kind = "Root")

    def fit(self, X, y):
        """Fit training data."""
        y = np.reshape(y, (len(y), 1))
        dataset = np.hstack((X, y))
        self.root.data = dataset
        self.recursion(self.root)

    def max_info_gain(self, dataset):
        """Calculate maximum information gain."""
        max_IG = -999
        for col in range(dataset.shape[1] - 1):
            for row in range(len(dataset)):
                branches = self.get_split(dataset, dataset[row, col], col)
                IG = self.gini_index(dataset)
                IG -= sum([self.gini_index(branch)*(len(branch)/len(dataset)) for
                        branch in branches])
                if IG > max_IG:
                    max_IG = IG
                    max_IG_position = (row, col)
        return dataset[max_IG_position], max_IG_position[1]

    def gini_index(self, dataset):
        """Calculate gini index for a dataset."""
        if dataset.size == 0:
            return 0
        labels, counts = np.unique(dataset[:, -1], return_counts = True)
        n_instances = dataset.shape[0]
        gini = 0.0
        for i in range(len(counts)):
            p = counts[i] / n_instances
            gini += p**2
        return 1 - gini

    def get_split(self, dataset, split_value, split_column):
        """Split the node into two branches."""
        n_columns = dataset.shape[1]
        left = list()
        right = list()
        for row in range(len(dataset)):
            if dataset[row, split_column] >= split_value:
                right.append(dataset[row])
            else:
                left.append(dataset[row])
        left = np.asarray(left)
        right = np.asarray(right)
        return left, right

    def is_leaf(self, node):
        """Check whether the current node is a leaf or not."""
        if (node.depth >= self.max_depth or len(np.unique(node.data[:, -1]))
                == 1 or len(node.data) < self.min_samples_split):
            return True
        question = self.max_info_gain(node.data)
        left, right = self.get_split(node.data, question[0], question[1])
        if (len(left) < self.min_samples_leaf or len(right) <
                self.min_samples_leaf):
            return True
        else:
            node.question = question
            node.left = Node(data = left, depth = node.depth + 1)
            node.right = Node(data = right, depth = node.depth + 1)
            return False

    def recursion(self, node):
        """Build the tree recursively."""
        #Return Condition.
        if self.is_leaf(node):
            values, counts = np.unique(node.data[:, -1], return_counts = True)
            node.label = int(values[np.argmax(counts)])
            node.kind = "Leaf"
            return

        #Build the tree.
        self.recursion(node.right)
        self.recursion(node.left)

    def predict(self, X):
        """Predict class label for new instance."""
        prediction = list()
        for row in range(len(X)):
            node = self.root
            while True:
                if node.kind == "Leaf":
                    prediction.append(node.label)
                    break
                if X[row, node.question[1]] >= node.question[0]:
                    node = node.right
                else:
                    node = node.left
        return np.asarray(prediction)
