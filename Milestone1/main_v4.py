import numpy as np
import sklearn.datasets
import pandas as pd
from abc import ABC, abstractmethod
import logging
import pickle



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
                 
    # X es la matriz de datos
    @property
    def _get_num_samples(self):
        self._num_samples = self.X.shape[0] #shape gives you the dimension of the array (number of rows)
        return self._num_samples
    
    @property
    def _get_num_features(self):
        self._num_features = self.X.shape[1] #number of columns
        return self._num_features

    def random_sampling(self, ratio_samples):
        # sample a subset of the dataset with replacement using
        # np.random.choice() to get the indices of rows in X and y
        # this function divides the initial dataset into two subsets: one for training and one for testing
        idx = np.random.choice(range(self.num_samples), int(self.num_samples*ratio_samples), replace=True)
        return DataSet(self.X[idx], self.y[idx])
    
    def split(self, idx, val): # divide the dataset into two subsets based on the value of a feature (umbral)
        left_idx = self.X[:, idx] < val
        right_idx = self.X[:, idx] >= val
        left_dataset = DataSet(self.X[left_idx], self.y[left_idx])
        right_dataset = DataSet(self.X[right_idx], self.y[right_idx])
        return left_dataset, right_dataset
    
    def most_frequent_label(self): # most frequent class in dataset
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
  
    def fit(self, X, y):
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = DataSet(X,y)
        self._make_decision_trees(dataset)
    
    def _make_decision_trees(self, dataset):
        self.trees = []
        logging.info('Creating Forest...\n')
        for i in range(self.num_trees):
            # sample a subset of the dataset with replacement using
            # np.random.choice() to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)  # the root of the decision tree
            self.trees.append(tree)
            logging.info(str(i+1)+' Tree created\n')
    
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
        logging.info('Most frequent label: %s', dataset.most_frequent_label())
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
            np.inf, np.inf, np.inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset) # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, \
                      best_split = idx, val, cost, [left_dataset, right_dataset]
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset): 
        # J(k,v) = (n_l/n)*G_l + (n_r/n)*G_r
        total = left_dataset.num_samples + right_dataset.num_samples #total number of samples
        if total == 0:
            logging.warning('Total number of samples is zero') 

        cost = abs(left_dataset.num_samples/total)*self.criterion.method(left_dataset) + \
                abs(right_dataset.num_samples/total)*self.criterion.method(right_dataset)  
        return cost

    def predict(self, X):
        ypred=[]
        for x in X:
            predictions=[root.predict(x) for root in self.trees]
            ypred.append(max(set(predictions), key=predictions.count))
        return np.array(ypred)

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
            p_c=np.sum(dataset.y==c)/dataset.num_samples
            gini -= (p_c)**2
        return gini 

class Entropy(Criterion):
    def method(self, dataset):
        #H(D)=-sum(p_c*log(p_c))
        C=len(np.unique(dataset.y))
        entropy=0
        for c in range(C):
            p_c=np.sum(dataset.y==c)/dataset.num_samples
            if p_c > 0:
                entropy -= p_c * np.log2(p_c)
        return entropy
        


# ---------------------------------------------------------- MAIN ----------------------------------------

def import_iris():
    iris = sklearn.datasets.load_iris()  # dictionary
    X, y = iris.data, iris.target  # array of x:150x4, array of y:150, 150 samples and 4 features
    return X, y

def import_sonar():
    df = pd.read_csv('./Milestone1/sonar.all-data.csv', header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y=='M').astype(int) # M = mine, R = rock
    return X, y

def import_mnist():

# Load the dataset
dataset=input(str("Enter the dataset you want to use (iris or sonar): "))
while(dataset!='iris' and dataset!='sonar'):
    dataset=input(str("Enter the dataset you want to use (iris or sonar): "))
if dataset=='iris':
    X, y = import_iris()
    logging.info('Dataset: IRIS')
else:
    X, y = import_sonar()
    logging.info('Dataset: SONAR')

# Train a random forest classifier
#Define the hyperparameters:
max_depth = 10      # maximum number of levels of a decision tree
min_size_split = 5  # if less, do not split a node
ratio_samples = 0.7 # sampling with replacement
num_trees = 10      # number of decision trees
num_features=X.shape[1]
num_random_features = int(np.sqrt(num_features)) # number of features to consider at # each node when looking for the best split

resposta=input(str("Enter the criterion you want to use (gini or entropy): "))
while(resposta!='gini' and resposta!='entropy'):
    resposta=input(str("Enter the criterion you want to use (gini or entropy): "))
if resposta=='gini':
    criterio=Gini()
    logging.info('Criterion: GINI')
else:
    criterio=Entropy()
    logging.info('Criterion: ENTROPY')

rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterio)
#Train the model
# train = make the decision trees
rf.fit(X,y) 
# classification           
ypred = rf.predict(X) 
# compute accuracy
num_samples_test = len(y)
num_correct_predictions = np.sum(ypred == y)
accuracy = num_correct_predictions/float(num_samples_test)
if float(num_samples_test)==0:
    logging.warning('Number of samples is zero')

print('\nAccuracy {} %\n'.format(100*np.round(accuracy,decimals=2)))
logging.info('Accuracy: %s \n', 100*np.round(accuracy,decimals=2))

logging.info('----- Script ended -----')
