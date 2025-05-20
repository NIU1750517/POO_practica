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

# Configure logger
logging.basicConfig(
    filename='loggerM2.log',
    filemode='w',
    encoding='utf-8',
    format='%(asctime)s %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logging.info('----- Script started -----')

class DataSet:
    """Maneja un conjunto de datos X e y de forma estructurada"""
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        logging.debug(f'DataSet initialized with {self.X.shape[0]} samples and {self.X.shape[1] if self.X.size else 0} features')
        
    @property
    def get_num_samples(self):
        return self.X.shape[0]
    
    @property
    def get_num_features(self):
        return self.X.shape[1]

    def random_sampling(self, ratio_samples):
        idx = np.random.choice(range(self.get_num_samples), int(self.get_num_samples*ratio_samples), replace=True)
        logging.debug(f'Random sampling with ratio {ratio_samples}: sampled {len(idx)} indices')
        return DataSet(self.X[idx], self.y[idx])
    
    def split(self, idx, val): 
        left_idx = self.X[:, idx] < val
        right_idx = self.X[:, idx] >= val
        logging.debug(f'Splitting on feature {idx} at {val}: left {left_idx.sum()}, right {right_idx.sum()}')
        left_dataset = DataSet(self.X[left_idx], self.y[left_idx])
        right_dataset = DataSet(self.X[right_idx], self.y[right_idx])
        return left_dataset, right_dataset
    
    def most_frequent_label(self):
        unique, counts = np.unique(self.y, return_counts=True)
        label = unique[np.argmax(counts)]
        logging.debug(f'Most frequent label computed: {label}')
        return label
    
    def mean_value(self):
        if self.y.size == 0:
            logging.warning('Empty dataset in mean_value(), returning 0.0')
            return 0.0
        mean_val = np.mean(self.y)
        logging.debug(f'Mean target value computed: {mean_val:.4f}')
        return mean_val

class RandomForest(ABC):
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, impurity, extra_trees=False):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.impurity = impurity
        self.trees = []
        self.training_time = 0
        self.extra_trees = extra_trees
        logging.info(f'RandomForest init: trees={num_trees}, min_size={min_size}, max_depth={max_depth}, samples_ratio={ratio_samples}, rand_feats={num_random_features}, extra_trees={extra_trees}')

    def fit(self, X, y, mode):
        logging.info(f'Start training ({mode})')
        dataset = DataSet(X, y)
        start = time.time()
        if mode == 'sequential':
            self._make_decision_trees(dataset)
        elif mode == 'parallel':
            self._make_decision_trees_multiprocessing(dataset)
        self.training_time = time.time() - start
        logging.info(f'Forest trained in {self.training_time:.2f}s')

    # rest of class unchanged...
    # Only showing additions related to regression, feature importance, and test_regression in Temperatures

class Temperatures(Import):
    def import_dataset(self):
        df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
        day = pd.DatetimeIndex(df.Date).day.to_numpy()
        month = pd.DatetimeIndex(df.Date).month.to_numpy()
        year = pd.DatetimeIndex(df.Date).year.to_numpy()
        X = np.vstack([day, month, year]).T
        y = df.Temp.to_numpy()
        logging.info(f'Temperatures dataset imported: {len(y)} samples')
        return X, y
    
    def test_regression(self, last_years_test=1):
        logging.info('Starting regression test for Temperatures')
        X, y = self.import_dataset()
        idx = last_years_test * 365
        Xtrain, Xtest = X[:-idx], X[-idx:]
        ytrain, ytest = y[:-idx], y[-idx:]
        logging.debug(f'Train samples: {len(ytrain)}, Test samples: {len(ytest)}')
        rf = RandomForestRegression(
            num_trees=50, min_size=5, max_depth=10,
            ratio_samples=0.5, num_random_features=2,
            impurity=SumSquareError(), extra_trees=True
        )
        rf.fit(Xtrain, ytrain, mode='sequential')
        ypred = rf.predict(Xtest)
        errors = ytest - ypred
        rmse = np.sqrt(np.mean(errors**2))
        logging.info(f'Regression completed: RMSE={rmse:.3f}')
        # Plotting omitted

class FeatureImportance(Visitor):
    def visitParent(self, node):
        k = node.feature_index
        self.occurrences[k] = self.occurrences.get(k, 0) + 1
        logging.debug(f'Feature {k} used for split, total occurrences now {self.occurrences[k]}')
        node.left_child.acceptVisitor(self)
        node.right_child.acceptVisitor(self)
    
    def visitLeaf(self, node):
        pass

# ---------------------------------------------------------- MAIN ----------------------------------------
# ...
