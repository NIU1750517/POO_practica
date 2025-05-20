import numpy as np
import sklearn.datasets
import pandas as pd
from abc import ABC, abstractmethod
import logging
import pickle
import time
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


logging.basicConfig(
    filename='loggerM2.log'
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
    """ This class handles a set of data X and Y (the target values) in a structured way"""
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
    """X is an information matrix"""         
    @property
    def get_num_samples(self):
        """Returns the number of rows, which would be the number of examples"""
        return self.X.shape[0] #Shape te da la dimensión de la matriz
    
    @property
    def get_num_features(self):
        """Returns the number of columns, which would be the number of features"""
        self._num_features = self.X.shape[1] 
        return self._num_features

    def random_sampling(self, ratio_samples):
        """Sample a subset of the dataset with replacement using np.random.choice() to get the row indices in X and Y"""
        idx = np.random.choice(range(self.get_num_samples), int(self.get_num_samples*ratio_samples), replace=True)
        return DataSet(self.X[idx], self.y[idx])
    
    def split(self, idx, val): 
        """Divides the dataset into two subsets based on the value of a feature (threshold)"""
        left_idx = self.X[:, idx] < val
        right_idx = self.X[:, idx] >= val
        left_dataset = DataSet(self.X[left_idx], self.y[left_idx])
        right_dataset = DataSet(self.X[right_idx], self.y[right_idx])
        return left_dataset, right_dataset
    
    def most_frequent_label(self):
        """Returns the target value (y) that occurs most frequently in the dataset"""
        unique, counts = np.unique(self.y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def mean_value(self):
        """Calculate the mean of class y"""
        if self.y.size == 0:  # Si no hay muestras
            logging.warning('Empty dataset in mean_value(), returning 0.0')
            return 0.0
        return np.mean(self.y)
    
class RandomForest(ABC):
    """Implements a random forest, a set of decision trees trained with different subsets of data and features"""
    def __init__(self,num_trees, min_size, max_depth, ratio_samples, num_random_features, impurity, extra_trees=False):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.impurity = impurity
        self.trees = []
        self.training_time = 0 
        self.extra_trees = extra_trees 


    def fit(self, X, y, mode):
        """Train the decision tree forest using the dataset"""
        dataset = DataSet(X,y)
        if mode=='sequential':
            self._make_decision_trees(dataset)
        elif mode=='parallel':
            self._make_decision_trees_multiprocessing(dataset)
     
    def _make_decision_trees(self, dataset):
        """Train all trees one by one in sequential mode"""
        self.trees = []
        logging.info('Creating Forest...\n')
        t1 = time.time()
        for i in tqdm(range(self.num_trees), desc="Creating trees...", unit=" tree"):
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)  # la raíz del árbol de decisión
            self.trees.append(tree)
            logging.info(str(i+1)+' Tree created\n')
        t2 = time.time()
        logging.debug('Training completed in %.2f seconds (%.4f sec/tree)', t2-t1, (t2-t1)/self.num_trees)
    
    
    def _make_node(self, dataset, depth):
        """Creates a decision tree node. This node can be a parent node or a child node depending on the conditions."""
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
    
    def _make_parent_or_leaf(self, dataset, depth):
        """Selects a random subset of features to make trees more diverse"""
        idx_features = np.random.choice(range(dataset.get_num_features),
                                        self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.get_num_samples > 0 or right_dataset.get_num_samples > 0
        if left_dataset.get_num_samples == 0 or right_dataset.get_num_samples == 0:
            logging.info('Leaf created')       
            """This is a special case: the dataset has samples from at least two classes, but the best split found
            doesn't separate the data properly (they all end up on one side of the split, leaving the other empty).
            So, instead of continuing to split, we create a new sheet directly."""
            return self._make_leaf(dataset)
        else:
            logging.info('Parent created')       
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _best_split(self, dataset):
        """Finds the best split (pair: feature, threshold) that best separates the data for a node in the tree by 
        exploring all possible splits"""
        best_feature, best_thresh, best_cost = None, None, float('inf')
        features = np.random.choice(dataset.get_num_features, self.num_random_features, False)
        
        for idx in features:
            if self.extra_trees:
                """Generates a single random threshold between the minimum and maximum feature values"""
                min_val = np.min(dataset.X[:, idx])
                max_val = np.max(dataset.X[:, idx])
                current_values = [np.random.uniform(min_val, max_val)]
            else:
                """Generates 10 quantiles of the characteristic"""
                current_values = np.quantile(dataset.X[:, idx], np.linspace(0.1, 0.9, 10))
            for val in current_values:
                left, right = dataset.split(idx, val)
                cost = self._CART_cost(left, right)
                if cost < best_cost:
                    best_feature, best_thresh, best_cost = idx, val, cost
                    best_split = (left, right)
        return best_feature, best_thresh, best_cost, best_split

    def _CART_cost(self, left, right):
        """Calculate the cost of a split to decide if it is a good option"""
        # J(k,v) = (n_l/n)*G_l + (n_r/n)*G_r
        total = left.get_num_samples + right.get_num_samples #número total de muestras
        if total == 0:
            logging.warning('Total number of samples is zero') 

        cost = abs(left.get_num_samples/total)*self.impurity.compute(left) + \
                abs(right.get_num_samples/total)*self.impurity.compute(right)  
        return cost
    
    def predict(self, X):
        """ Predict the target value for each row of X, having each tree vote, and choose the class with the most votes"""
        ypred=[]
        for x in tqdm(X, desc="Predicting trees...", unit=" row"):
            predictions=[root.predict(x) for root in self.trees]
            combined = self._combinePredictions(predictions)
            ypred.append(combined)
        return np.array(ypred)
        
    def _target(self, dataset, nproc):
        """Creates a single tree that trains alongside other trees at the same time"""
        logging.debug('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)
        logging.debug('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset):
        """Train multiple trees at the same time, using multiple cores"""
        logging.info('Creating Forest with multiprocessing...\n')
        t1 = time.time()
        with multiprocessing.Pool() as pool:
            args = [(dataset, nproc) for nproc in range(self.num_trees)]
            self.trees = list(tqdm(
                        pool.starmap(self._target, args),
                        total=self.num_trees,
                        desc="Creating trees...",
                        unit=" tree"
                    ))
        t2 = time.time()
        logging.debug('Parallel training completed in %.2f seconds (%.2f sec/tree)', t2-t1, (t2-t1)/self.num_trees)

    def feature_importance(self):
        """Calculate the importance of each feature based on how many times it has been used to split nodes in all trees in the forest"""
        feat_imp_visitor = FeatureImportance()
        for tree in self.trees:
            tree.acceptVisitor(feat_imp_visitor) 
        return feat_imp_visitor.occurrences

    def print_trees(self):
        """Creates a text file in which the representation of all the trees in the forest is saved"""
        filename = 'decisiontrees.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            for i, tree in enumerate(self.trees):
                f.write(f"decision tree {i+1}\n")
                tree_printer = PrinterTree(f)
                tree.acceptVisitor(tree_printer)
        f.close()
        print(f"File '{filename}' has been successfully exported!!\n")
 

    def other_method_traversing_trees():
        pass

    @abstractmethod
    def _combinePredictions(self, predictions):
        pass

    @abstractmethod
    def _make_leaf(self, dataset): 
        pass

class RandomForestClassifier(RandomForest):
    @staticmethod
    def _combinePredictions(predictions):
        """Combine the predictions of all trees using majority voting"""
        return np.argmax(np.bincount(predictions))

    def _make_leaf(self, dataset): 
        """Creates a leaf of the tree, returning the most frequent class in that data group (label)"""  
        logging.info('Leaf created') 
        logging.info('Most frequent label: %s', dataset.most_frequent_label())
        return Leaf(dataset.most_frequent_label())
    
class RandomForestRegression(RandomForest):
    @staticmethod
    def _combinePredictions(predictions):
        """Combine the predictions of all the trees by taking the average"""
        return np.mean(predictions)

    def _make_leaf(self, dataset): 
        """Creates a leaf of the tree, returning the mean value of the labels (the target values, y) in that data set"""  
        logging.info('Leaf created') 
        logging.info('Most frequent label: %s', dataset.mean_value())
        return Leaf(dataset.mean_value())    


 
class Node(ABC):
    """Abstract class that serves as a template for other node types"""
    def __init__(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def acceptVisitor(self, v):
        pass

class Leaf(Node):
    """Represents a node of the tree (it is a leaf)"""
    def __init__(self, label):
        self.label = label

    def predict(self, X):
        """Always returns the label, regardless of the value of X, because no further decisions are made on a sheet."""
        return self.label
    
    def acceptVisitor(self, v):
        v.visitLeaf(self)

class Parent(Node):
    """Represents a node in the tree that makes decisions based on a feature and a threshold"""
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold # umbral

    def predict(self, X):
        """Check the value of X at position feature_index and make a prediction"""
        if X[self.feature_index]<self.threshold:
            return self.left_child.predict(X)
        else: 
            return self.right_child.predict(X) 

    def acceptVisitor(self, v):
        """Allows a visitor to interact with this node"""
        v.visitParent(self)

class ImpurityMeasure(ABC):
    """Abstract class that defines a criterion to measure how good a split is"""
    @abstractmethod
    def compute(self, dataset):
        pass

class Gini(ImpurityMeasure):
    """Criterion that calculates the Gini index of a dataset"""
    def compute(self, dataset):
        #G(D)=1-sum(p_c^2)
        C=len(np.unique(dataset.y))
        gini=1
        for c in range(C):
            p_c=np.sum(dataset.y==c)/dataset.get_num_samples
            gini -= (p_c)**2
        return gini 

class Entropy(ImpurityMeasure):
    """Criterion that calculates the entropy of a dataset"""
    def compute(self, dataset):
        #H(D)=-sum(p_c*log(p_c))
        C=len(np.unique(dataset.y))
        entropy=0
        for c in range(C):
            p_c=np.sum(dataset.y==c)/dataset.get_num_samples
            if p_c > 0:
                entropy -= p_c * np.log2(p_c)
        return entropy
    
class SumSquareError(ImpurityMeasure):
    """Calculates the SSE for a given dataset"""
    def compute(self, dataset):
        y = dataset.y
        mean_y = dataset.mean_value()  # Usar método seguro de DataSet
        sse = np.sum((y - mean_y) ** 2)
        return sse
    
class Import(ABC):
    """Abstract class that establishes a structure for importing and dividing datasets into training and test subsets"""
    @abstractmethod
    def import_dataset(self):
        pass
    def divide_dataset(self,X, y):
        """Splits the dataset in a specific way. (70% training, 30% test)"""
        ratio_train, ratio_test = 0.7, 0.3 
        num_samples, num_features = X.shape # 150, 4
        idx = np.random.permutation(range(num_samples))
        """shuffle {0,1, ... 149} because the samples are sorted by class!"""
        num_samples_train = int(num_samples*ratio_train)
        num_samples_test = int(num_samples*ratio_test)
        idx_train = idx[:num_samples_train]
        idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        return X_train, y_train, X_test, y_test
    
    @abstractmethod
    def test_occurrences(self, rf):
        pass

class Iris(Import):
    def import_dataset(self):
        """Loads the iris dataset and splits it into X and Y and then divides it into training and test sets using the
          divide_dataset function."""
        iris = sklearn.datasets.load_iris()  # dictonary
        X, y = iris.data, iris.target  # array of x:150x4, array of y:150, 150 samples and 4 features
        X_train,y_train,X_test,y_test = self.divide_dataset(X,y)
        return X_train, y_train, X_test, y_test
    
    def test_occurrences(self, rf):
        """Calculates and visualizes the frequency of use of each feature in a random forest model.
        Uses the 'feature_importance' method to calculate how many times each feature has been used in decision trees.
        Displays the results in a bar chart."""
        occurrences = rf.feature_importance()
        print('Iris occurrences for {} trees:'.format(rf.num_trees))
        print("\t", occurrences)
        counts = np.array(list(occurrences.items()))
        plt.figure(), plt.bar(counts[:, 0], counts[:, 1])
        plt.xlabel('feature')
        plt.ylabel('occurrences')
        plt.title('Iris feature importance\n{} trees'.format(rf.num_trees))
        plt.show()

class Sonar(Import):
    def import_dataset(self):
        """Load the Sonar dataset, split the data into X for the columns containing the features (all but the last) and 
        Y for the last column containing the target values. Transform the target values ​​to integers and split the data into training and test sets."""
        df = pd.read_csv('./Milestone1/sonar.all-data.csv', header=None)
        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy(dtype=str)
        y = (y=='M').astype(int) # M = mina, R = roca
        X_train,y_train,X_test,y_test = self.divide_dataset(X,y)
        return X_train, y_train, X_test, y_test
    
    def test_occurrences(self, rf):
        """Calculates and visualizes the frequency of use of each feature in a random forest model.
        Uses the 'feature_importance' method to calculate how many times each feature has been used in decision trees.
        Displays the results in a bar chart."""
        occurrences = rf.feature_importance() # a dictionary
        counts = np.array(list(occurrences.items()))
        plt.figure(), plt.bar(counts[:, 0], counts[:, 1])
        plt.xlabel('feature')
        plt.ylabel('occurrences')
        plt.title(' feature importance\n{} trees'.format(rf.num_trees))
        plt.show()

class Mnist(Import):
    def import_dataset(self):
        """Opens and loads the contents of a pickle file, and directly returns the already separated training and test sets"""
        with open("./Milestone1/mnist.pkl",'rb') as file:
            mnist = pickle.load(file)
        Xtrain, ytrain, Xtest, ytest = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
        return Xtrain, ytrain, Xtest, ytest
    
    def test_occurrences(self, rf):
        """Calculates and visualizes the frequency of use of each feature in a random forest model.
        Uses the 'feature_importance' method to obtain the number of times each feature has been used in decision trees.
        Displays the results as an image with a color scale, where each pixel represents a feature."""
        occurrences = rf.feature_importance()
        ima = np.zeros(28*28)
        for k in occurrences.keys():
            ima[k] = occurrences[k]
        plt.figure()
        plt.imshow(np.reshape(ima,(28,28)))
        plt.colorbar()
        plt.title('Feature importance MNIST')
        plt.show()        
        
class Temperatures(Import):
    def import_dataset(self):
        """Loads the dataset of daily minimum temperatures recorded in Melbourne, Australia (1981–1990).
        Units are in degrees Celsius.
        Converts the date column into three separate features: day, month, and year.
        The daily minimum temperature is used as the target variable (y)."""
        
        df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/'
        'Datasets/master/daily-min-temperatures.csv')
        #These are the features to regress:
        day = pd.DatetimeIndex(df.Date).day.to_numpy() # 1...31
        month = pd.DatetimeIndex(df.Date).month.to_numpy() # 1...12
        year = pd.DatetimeIndex(df.Date).year.to_numpy() # 1981...1999
        X = np.vstack([day, month, year]).T # np array of 3 columns
        y = df.Temp.to_numpy()
        return X, y
    
    def test_occurrences(self, rf):
        pass

    
    def test_regression(self, last_years_test=1):
        """Train and evaluate a random forest model for regression using the temperature dataset"""
        X, y = self.import_dataset()
        plt.plot(y,'.-')
        plt.xlabel('day in 10 years'), plt.ylabel('min. daily temperature')
        idx = last_years_test*365
        Xtrain = X[:-idx,:] # first years
        Xtest = X[-idx:]
        ytrain = y[:-idx] # last years
        ytest = y[-idx:]
        #
        rf = RandomForestRegression(num_trees=50, min_size=5, max_depth=10, ratio_samples=0.5, num_random_features=2, impurity=SumSquareError(), extra_trees=True)        
        rf.fit(Xtrain, ytrain, mode='sequential')
        ypred=  rf.predict(Xtest)
        #
        plt.figure()
        x = range(idx)
        for t, y1, y2 in zip(x, ytest, ypred):
            plt.plot([t, t], [y1, y2], 'k-')
        plt.plot([x[0], x[0]],[ytest[0], ypred[0]], 'k-', label='error')
        plt.plot(x, ytest, 'g.', label='test')
        plt.plot(x, ypred, 'y.', label='prediction')
        plt.xlabel('day in last {} years'.format(last_years_test))
        plt.ylabel('min. daily temperature')
        plt.legend()
        errors = ytest - ypred
        rmse = np.sqrt(np.mean(errors**2))
        plt.title('root mean square error : {:.3f}'.format(rmse))
        plt.show()


class Visitor(ABC):
    @abstractmethod
    def visitParent(self, node):
        pass

    @abstractmethod
    def visitLeaf(self, node):
        pass


class FeatureImportance(Visitor):
    def __init__(self):
        """A dictionary is initialized to count the number of times each feature is used"""
        self.occurrences = {}

    def visitParent(self, node):
        """Method called when visiting a Parent node that records the feature used in that node
        and increments its dictionary counter.
        Then it recursively traverses the left and right children."""
        k = node.feature_index 
        self.occurrences[k] = self.occurrences.get(k, 0) + 1
        node.left_child.acceptVisitor(self)
        node.right_child.acceptVisitor(self)
    
    def visitLeaf(self, node):
        pass

class PrinterTree(Visitor):
    def __init__(self, file, depth=0):
        self._file = file
        self._depth = depth
    
    def visitParent(self, node):
        """Method called visiting a parent node.
        A line indicating the feature index and threshold used in the split is written to the file.
        Then it recursively traverses the left and right children, increasing the depth."""
        self._file.write('\t'*self._depth + 'parent, features indx. {}, threshold {:.2f}\n'.format(node.feature_index, node.threshold))
        self._depth += 1
        node.left_child.acceptVisitor(self)
        node.right_child.acceptVisitor(self)
        self._depth -= 1
    
    def visitLeaf(self, node):
        """Method called when visiting a leaf in the tree.
        Writes the target value (label) assigned to that leaf to the file."""
        self._file.write('\t'*self._depth + 'leaf, label {}\n'.format(node.label))

# ---------------------------------------------------------- MAIN ----------------------------------------

if __name__ == '__main__':
    print("|------------------------------------------------------------------------------|")

    type = input("\nCLASSIFIER or REGRESSION: ").lower()
    while type not in ['classifier', 'regression']:
        type = input("Invalid. Choose (CLASSIFIER/REGRESSION): ").lower()
    if type=='regression':
        print("\nTemperatures Dataset imported.")
        print("SumSquareError selected.\n")
        temp=Temperatures()
        temp.test_regression()

    else:
        dataset = input("\nDataset (iris/sonar/mnist): ").lower()    
        while dataset not in ['iris', 'sonar', 'mnist']:
            dataset = input("Invalid dataset. Choose (iris/sonar/mnist): ").lower()
        if dataset=='iris':
            dataset=Iris()
            X_train, y_train, X_test, y_test = dataset.import_dataset()
            logging.info('Dataset: IRIS')
        elif dataset=='sonar':
            dataset=Sonar()
            X_train, y_train, X_test, y_test = dataset.import_dataset()
            logging.info('Dataset: SONAR')
        elif dataset=='mnist':
            dataset=Mnist()
            X_train, y_train, X_test, y_test = dataset.import_dataset()
            logging.info('Dataset: MNIST')

        impurity = input("ImpurityMeasure (gini/entropy): ").lower()
        while impurity not in ['gini', 'entropy']:
            impurity = input("Invalid impurity measure. Choose (gini/entropy): ").lower()
        if impurity == 'gini':
            impurity=Gini()
            logging.info('Impurity: GINI')
        elif impurity == 'entropy':
            impurity=Entropy()
            logging.info('Impurity: ENTROPY')

        extra_trees = input("Use Extra-Trees optimization? (yes/no): ").lower().strip() == 'yes'

        mode=input(str("Mode (sequential/parallel): ")).lower()
        while mode not in ['sequential', 'parallel']:        
            mode = input("Invalid mode. Choose (sequential/parallel): ").lower()
        print("\n")
        #Define the hyperparameters:
        max_depth = 10    # Maximum number of levels in a decision tree
        min_size_split = 25  # If it is smaller, do not split a node
        ratio_samples = 1 # Sampling with replacement
        num_trees = 50     #Number of decision trees
        multiprocessing.cpu_count() == 8
        num_features=X_train.shape[1]
        num_random_features = int(np.sqrt(num_features)) #Number of features to consider at each node when searching for the best split
        
        time_start=time.time()
        rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, impurity, extra_trees)
        #Train the model
        #train = make the decision trees
        rf.fit(X_train, y_train, mode) 

        # classification           
        ypred = rf.predict(X_test) 
        # compute accuracy
        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy = num_correct_predictions/float(num_samples_test)
        if float(num_samples_test)==0:
            logging.warning('Number of samples is zero')

        print('\n\nAccuracy {} %'.format(100*np.round(accuracy,decimals=2)))
        logging.info('Accuracy: %.2f %%', 100*np.round(accuracy, decimals=2))
        time_end = time.time()  # <-- Stop TOTAL timer
        total_time = time_end-time_start
        
        print(f'Total Time: {int(total_time // 60)}min {(total_time % 60):.4f}s\n\n')  # Format MM min SS sec
        logging.info('Total Time: %d min %.4fs\n', int(total_time // 60), (total_time % 60))

        rf.print_trees()
        dataset.test_occurrences(rf)

        
        print("\n|------------------------------------------------------------------------------|")
        logging.info('----- Script ended -----')

            


            


