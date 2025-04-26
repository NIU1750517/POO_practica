import numpy as np
import sklearn.datasets
import pandas as pd
from abc import ABC, abstractmethod
import logging
import pickle
import time
import multiprocessing

logging.basicConfig(
    filename='loggerM1.log'
    , # output to file
    filemode='w'
    , # rewrite at each execution, don't append
    encoding='utf-8'
    , # also non Ascii characters
    format='%(asctime)s %(levelname)s - %(message)s '
    , # print date and time also
    level=logging.DEBUG # default level
)
logging.info('----- Script started -----\n')

class DataSet:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
      # X is the info matrix           
    @property
    def get_num_samples(self):
        return self.X.shape[0] #shape gives you the dimension of the array (number of rows)
    
    @property
    def get_num_features(self):
        self._num_features = self.X.shape[1] #number of columns
        return self._num_features

    def random_sampling(self, ratio_samples):
        # sample a subset of the dataset with replacement using
        # np.random.choice() to get the indices of rows in X and y
        # this function divides the initial dataset into two subsets: one for training and one for testing
        idx = np.random.choice(range(self.get_num_samples), int(self.get_num_samples*ratio_samples), replace=True)
        return DataSet(self.X[idx], self.y[idx])
    
    def split(self, idx, val): # divide the dataset into two subsets based on the value of a feature (umbral)
        left_idx = self.X[:, idx] < val
        right_idx = self.X[:, idx] >= val
        left_dataset = DataSet(self.X[left_idx], self.y[left_idx])
        right_dataset = DataSet(self.X[right_idx], self.y[right_idx])
        return left_dataset, right_dataset
    def most_frequent_label(self):
        unique, counts = np.unique(self.y, return_counts=True)
        return unique[np.argmax(counts)]

class RandomForestClassifier:
    def __init__(self,num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.criterion = criterion
        self.trees = []
  
    def fit(self, X, y, mode):
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = DataSet(X,y)
        if mode=='sequencial':
            self._make_decision_trees(dataset)
        elif mode=='parallel':
            self._make_decision_trees_multiprocessing(dataset)
    
    def _make_decision_trees(self, dataset):
        self.trees = []
        logging.info('Creating Forest...\n')
        for i in range(self.num_trees):
            # sample a subset of the dataset with replacement using
            # np.random.choice() to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)
            self.trees.append(tree)
            logging.info(str(i+1)+' Tree created\n')
    
    def _make_node(self, dataset, depth):
        logging.info('Making node...')
        if (depth >= self.max_depth 
            or dataset.get_num_samples <= self.min_size 
            or np.unique(dataset.y).size == 1):
            return self._make_leaf(dataset)
        
        feature_idx, threshold, _, (left, right) = self._best_split(dataset)
        
        if left.get_num_samples == 0 or right.get_num_samples == 0:
            return self._make_leaf(dataset)
            
        node = Parent(feature_idx, threshold)
        node.left_child = self._make_node(left, depth + 1)
        node.right_child = self._make_node(right, depth + 1)
        return node

    def _make_leaf(self, dataset):       
        logging.info('Leaf created')       
        #label = most frequent class in dataset
        logging.info('Most frequent label: %s', dataset.most_frequent_label())
        return Leaf(dataset.most_frequent_label())
    def _make_parent_or_leaf(self, dataset, depth):
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.get_num_features),
                                        self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.get_num_samples > 0 or right_dataset.get_num_samples > 0
        if left_dataset.get_num_samples == 0 or right_dataset.get_num_samples == 0:
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

    def _best_split(self, dataset):
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature, best_thresh, best_cost = None, None, float('inf')
        features = np.random.choice(dataset.get_num_features, self.num_random_features, False)
        
        for idx in features:
            values = np.quantile(dataset.X[:, idx], np.linspace(0.1, 0.9, 10))
            for val in values:
                left, right = dataset.split(idx, val)
                cost = self._CART_cost(left, right)
                if cost < best_cost:
                    best_feature, best_thresh, best_cost = idx, val, cost
                    best_split = (left, right)
        return best_feature, best_thresh, best_cost, best_split

    def _CART_cost(self, left, right):
       # J(k,v) = (n_l/n)*G_l + (n_r/n)*G_r
        total = left.get_num_samples + right.get_num_samples #total number of samples
        if total == 0:
            logging.warning('Total number of samples is zero') 

        cost = abs(left.get_num_samples/total)*self.criterion.method(left) + \
                abs(right.get_num_samples/total)*self.criterion.method(right)  
        return cost
    def predict(self, X):
            ypred=[]
            for x in X:
                predictions=[root.predict(x) for root in self.trees]
                ypred.append(max(set(predictions), key=predictions.count))
            return np.array(ypred)
        
    def _target(self, dataset, nproc):
        logging.debug('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)
        logging.debug('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset):
        logging.info('Creating Forest with multiprocessing...\n')
        t1 = time.time()
        with multiprocessing.Pool() as pool:
            args = [(dataset, nproc) for nproc in range(self.num_trees)]
            self.trees = pool.starmap(self._target, args)
        # use pool.map instead if only one argument for _target
        t2 = time.time()
        logging.debug('Parallel training completed in %.2f seconds (%.2f sec/tree)', t2-t1, (t2-t1)/self.num_trees)


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
        self.threshold = threshold # umbral

    def predict(self, X):
        if X[self.feature_index]<self.threshold:
            return self.left_child.predict(X)
        else:
            return self.right_child.predict(X)

class Criterion(ABC):
    @abstractmethod
    def method(self, dataset):
        pass

class Gini(Criterion):
    def method(self, dataset):
        #G(D)=1-sum(p_c^2)
        C=len(np.unique(dataset.y))
        gini=1
        for c in range(C):
            p_c=np.sum(dataset.y==c)/dataset.get_num_samples
            gini -= (p_c)**2
        return gini 

class Entropy(Criterion):
    def method(self, dataset):
        #H(D)=-sum(p_c*log(p_c))
        C=len(np.unique(dataset.y))
        entropy=0
        for c in range(C):
            p_c=np.sum(dataset.y==c)/dataset.get_num_samples
            if p_c > 0:
                entropy -= p_c * np.log2(p_c)
        return entropy
class Import(ABC):
    @abstractmethod
    def import_dataset(self):
        pass
    def divide_dataset(self,X, y):
        ratio_train, ratio_test = 0.7, 0.3 # 70% train, 30% test
        num_samples, num_features = X.shape # 150, 4
        idx = np.random.permutation(range(num_samples))
        # shuffle {0,1, ... 149} because samples come sorted by class!
        num_samples_train = int(num_samples*ratio_train)
        num_samples_test = int(num_samples*ratio_test)
        idx_train = idx[:num_samples_train]
        idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        return X_train, y_train, X_test, y_test

class Iris(Import):
    def import_dataset(self):
        iris = sklearn.datasets.load_iris()  # dictionary
        X, y = iris.data, iris.target  # array of x:150x4, array of y:150, 150 samples and 4 features
        X_train,y_train,X_test,y_test = self.divide_dataset(X,y)
        return X_train, y_train, X_test, y_test
class Sonar(Import):
    def import_dataset(self):
        df = pd.read_csv('./Milestone1/sonar.all-data.csv', header=None)
        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy(dtype=str)
        y = (y=='M').astype(int) # M = mine, R = rock
        X_train,y_train,X_test,y_test = self.divide_dataset(X,y)
        return X_train, y_train, X_test, y_test
class Mnist(Import):
    def import_dataset(self):
        with open("./Milestone1/mnist.pkl",'rb') as file:
            mnist = pickle.load(file)
        Xtrain, ytrain, Xtest, ytest = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
        return Xtrain, ytrain, Xtest, ytest




# ---------------------------------------------------------- MAIN ----------------------------------------

if __name__ == '__main__':
    # Load the dataset
    dataset = input("Dataset (iris/sonar/mnist): ").lower()    
    while dataset not in ['iris', 'sonar', 'mnist']:
        dataset = input("Invalid dataset. Choose (iris/sonar/mnist): ").lower()
    if dataset=='iris':
        iris=Iris()
        X_train, y_train, X_test, y_test = iris.import_dataset()
        logging.info('Dataset: IRIS')
    elif dataset=='sonar':
        sonar=Sonar()
        X_train, y_train, X_test, y_test = sonar.import_dataset()
        logging.info('Dataset: SONAR')
    else:
        mnist=Mnist()
        X_train, y_train, X_test, y_test = mnist.import_dataset()
        logging.info('Dataset: MNIST')

    # Train a random forest classifier
    #Define the hyperparameters:
    max_depth = 10    # maximum number of levels of a decision tree
    min_size_split = 5  # if less, do not split a node
    ratio_samples = 0.7 # sampling with replacement
    num_trees = 10     # number of decision trees
    multiprocessing.cpu_count() == 8
    num_features=X_train.shape[1]
    num_random_features = int(np.sqrt(num_features)) # number of features to consider at # each node when looking for the best split

    criterion = input("Criterion (gini/entropy): ").lower()
    while criterion not in ['gini', 'entropy']:
        criterion = input("Invalid criterion. Choose (gini/entropy): ").lower()
    if criterion=='gini':
        criterio=Gini()
        logging.info('Criterion: GINI')
    else:
        criterio=Entropy()
        logging.info('Criterion: ENTROPY')

    mode=input(str("Mode (sequencial/parallel): ")).lower()
    while mode not in ['sequencial', 'parallel']:        
        mode = input("Invalid mode. Choose (sequencial/parallel): ").lower()

    rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterio)
    #Train the model
    # train = make the decision trees
    rf.fit(X_train, y_train, mode) 
    # classification           
    ypred = rf.predict(X_test) 
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    if float(num_samples_test)==0:
        logging.warning('Number of samples is zero')

    print('\nAccuracy {} %\n'.format(100*np.round(accuracy,decimals=2)))
    logging.info('Accuracy: %s \n', 100*np.round(accuracy,decimals=2))

    logging.info('----- Script ended -----')
