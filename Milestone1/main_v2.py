import numpy as np
import sklearn.datasets
from abc import ABC, abstractmethod
import logging

logging.basicConfig(
    filename='loggerM1.log'
    , # output to file
    filemode='w'
    , # rewrite at each execution, don't append
    encoding='utf-8'
    , # also non Ascii characters
    format='%(asctime)s %(message)s'
    , # print date and time also
    level=logging.DEBUG # default level
)
logging.info('----- Script started -----')

iris = sklearn.datasets.load_iris()  # diccionario
print(iris.DESCR)
X, y = iris.data, iris.target  # array de x:150x4, array de y:150, 150 muestras y 4 features

class DataSet:
    def __init__(self, X, y, num_trees, ratio_samples):
        self.X = X
        self.y = y
        logging.debug('Dataset created')  

    # X es la matriz de datos
    @property
    def num_samples(self):
        self._num_samples = X.shape[0]
        return self._num_samples
    @property
    def num_features(self):
        self._num_features = X.shape[1]
        return self._num_features
        
    
class RandomForestClassifier:
    def __init__(self,num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion='gini'):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.criterion = criterion
        self.trees = []
  
    def fit(self, X, y):
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = DataSet(X,y)
        self._make_decision_trees(dataset)
    
    def _make_decision_trees(self, dataset):
        self.trees = []
        logging.info('Creating trees...')
        for i in range(self.num_trees):
            # sample a subset of the dataset with replacement using
            # np.random.choice() to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)  # the root of the decision tree
            self.decision_trees.append(tree)
    
    def _make_node(self, dataset, depth):
        logging.info('Making node...')
        if depth == self.max_depth \
                or dataset.num_samples <= self.min_size \
                or len(np.unique(dataset.y)) == 1:
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node

    def _make_leaf(self, dataset): 
        logging.info('Leaf created')       
        #label = most frequent class in dataset
        return Leaf(dataset.most_frequent_label())

    def _make_parent_or_leaf(self, dataset, depth):
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features),
                                        self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            logging.info('Leaf created')       
            # this is an special case : dataset has samples of at least two 
            # classes but the best split is moving all samples to the left or right
            # dataset and none to the other, so we make a leaf instead of a parent
            return self._make_leaf(dataset)
        else:
            logging.info('Parent created')       
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node
    
    def _best_split(self, idx_features, dataset):
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = \
            np.Inf, np.Inf, np.Inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset) # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, \
                      best_split = idx, val, cost, [left_dataset, right_dataset]
        logging.info('Best split found: %s', best_split)
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset): 
        # J(k,v) = (n_l/n)*G_l + (n_r/n)*G_r
        total = left_dataset.num_samples + right_dataset.num_samples
        cost = abs(left_dataset.num_samples/total)*self._gini(left_dataset) + \
               abs(right_dataset.num_samples/total)*self._gini(right_dataset)
        return cost

    def _gini(self, dataset):
        C=len(np.unique(dataset.y))
        gini=1
        for c in range(C):
            gini-= (np.sum(dataset.y==c)/dataset.num_samples)**2
        logging.info('Gini Purity: %s', gini)
        return gini 

class Node(ABC):
    def __init__(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child

    @abstractmethod
    def predict(self, X):
        pass

class Leaf(Node):
    def __init__(self, label):
        self.label = label

    def predict(self, X):
        return self.label

class Parent(Node):
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold

    def predict(self, X):
        return self.label
