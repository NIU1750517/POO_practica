import numpy as np
import sklearn.datasets
import pandas as pd
from abc import ABC, abstractmethod
import logging
import pickle
import time
import multiprocessing
from tqdm import tqdm


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
    #Esta clase maneja un conjunto de datos X e y (los valores objetivo) de forma estructurada
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
      # X es una matriz de informacion           
    @property
    def get_num_samples(self):
        # devuelve el número de filas, que sería el número de ejemplos
        return self.X.shape[0] #shape te da la dimensión de la matriz
    
    @property
    def get_num_features(self):
        #devuelve el número de columnas, que sería el número de caracteristicas
        self._num_features = self.X.shape[1] 
        return self._num_features

    def random_sampling(self, ratio_samples):
        # muestra un subconjunto del dataset con reemplazo usando
        # np.random.choice() para obtener los índices de las filas en X e y
        idx = np.random.choice(range(self.get_num_samples), int(self.get_num_samples*ratio_samples), replace=True)
        return DataSet(self.X[idx], self.y[idx])
    
    def split(self, idx, val): 
        # divide el dataset en dos subconjuntos basandose en el valor de una caracterÍstica (umbral)
        left_idx = self.X[:, idx] < val
        right_idx = self.X[:, idx] >= val
        left_dataset = DataSet(self.X[left_idx], self.y[left_idx])
        right_dataset = DataSet(self.X[right_idx], self.y[right_idx])
        return left_dataset, right_dataset
    
    def most_frequent_label(self):
        #devuelve el valor objetivo (y) que aparece con más frecuencia en el dataset
        unique, counts = np.unique(self.y, return_counts=True)
        return unique[np.argmax(counts)]

class RandomForestClassifier:
    #implementa un random forest,, un conjunto de árboles de decisión entrenados con diferentes subconjuntos de datos y características
    def __init__(self,num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion, extra_trees=False):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.criterion = criterion
        self.trees = []
        self.training_time = 0 
        self.extra_trees = extra_trees

    def fit(self, X, y, mode):
        #Entrena el bosque de árboles de decisión usando el conjunto de datos
        dataset = DataSet(X,y)
        if mode=='sequencial':
            self._make_decision_trees(dataset)
        elif mode=='parallel':
            self._make_decision_trees_multiprocessing(dataset)
     
    def _make_decision_trees(self, dataset):
        #Entrena todos los árboles uno por uno en modo secuencial
        self.trees = []
        logging.info('Creating Forest...\n')
        for i in tqdm(range(self.num_trees), desc="Creating trees...", unit=" tree"):
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)  # la raíz del árbol de decisión
            self.trees.append(tree)
            logging.info(str(i+1)+' Tree created\n')
    
    def _make_node(self, dataset, depth):
        #Crea un nodo del árbol de decisión. Este nodo puede ser un nodo padre o un nodo hijo dependiendo de las condiciones
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
        #Crea una hoja del árbol , devolviendo la clase mas frecuente en ese grupo de datos (label)  
        logging.info('Leaf created') 
        logging.info('Most frequent label: %s', dataset.most_frequent_label())
        return Leaf(dataset.most_frequent_label())
    
    def _make_parent_or_leaf(self, dataset, depth):
        # Selecciona un subconjunto aleatorio de carateristicas para hacer que los árboles sean más diversos.
        idx_features = np.random.choice(range(dataset.get_num_features),
                                        self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.get_num_samples > 0 or right_dataset.get_num_samples > 0
        if left_dataset.get_num_samples == 0 or right_dataset.get_num_samples == 0:
            logging.info('Leaf created')       
            # Este es un caso especial : dataset tiene muestras de al menos dos
            # clases pero la mejor división encontrada no separa los datos correctamente 
            # (todos terminan en un solo lado del split, dejando el otro vacío).
            # Por lo que, en vez de seguir dividiendo, creamos directamente una hoja.
            return self._make_leaf(dataset)
        else:
            logging.info('Parent created')       
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _best_split(self, dataset):
        # Encuentra el mejor split (par: característica, umbral) que separa mejor los datos para un nodo del árbol explorando todos los splits posibles
        best_feature, best_thresh, best_cost = None, None, float('inf')
        features = np.random.choice(dataset.get_num_features, self.num_random_features, False)
        
        for idx in features:
            if self.extra_trees:
                # Genera un único umbral aleatorio entre el mínimo y el máximo valor de la característica
                min_val = np.min(dataset.X[:, idx])
                max_val = np.max(dataset.X[:, idx])
                current_values = [np.random.uniform(min_val, max_val)]
            else:
                # Genera 10 cuantiles de la característica
                current_values = np.quantile(dataset.X[:, idx], np.linspace(0.1, 0.9, 10))
            for val in current_values:
                left, right = dataset.split(idx, val)
                cost = self._CART_cost(left, right)
                if cost < best_cost:
                    best_feature, best_thresh, best_cost = idx, val, cost
                    best_split = (left, right)
        return best_feature, best_thresh, best_cost, best_split

    def _CART_cost(self, left, right):
        # Calcula el costo de un split para decidir si es bueno
        # J(k,v) = (n_l/n)*G_l + (n_r/n)*G_r
        total = left.get_num_samples + right.get_num_samples #número total de muestras
        if total == 0:
            logging.warning('Total number of samples is zero') 

        cost = abs(left.get_num_samples/total)*self.criterion.method(left) + \
                abs(right.get_num_samples/total)*self.criterion.method(right)  
        return cost
    
    def predict(self, X):
        #Predice el valor objetivo para cada fila de X, haciendo que cada árbol vote, y elige la clase con más votos
        ypred=[]
        for x in tqdm(X, desc="Predicting trees...", unit=" row"):
            predictions=[root.predict(x) for root in self.trees]
            ypred.append(max(set(predictions), key=predictions.count))
        return np.array(ypred)
        
    def _target(self, dataset, nproc):
        #Crea un solo árbol que se entrena junto a otros árboles al mismo tiempo
        logging.debug('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)
        logging.debug('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset):
        #Entrena varios árboles al mismo tiempo, usando múltiples núcleos
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
        
        t2 = time.time()
        logging.debug('Parallel training completed in %.2f seconds (%.2f sec/tree)', t2-t1, (t2-t1)/self.num_trees)


class Node(ABC):
    #clase abstracta que sirve como plantilla para otros tipos de nodo
    def __init__(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child

    @abstractmethod
    def predict(self, X):
        pass

class Leaf(Node):
    #representa un nodo del árbol (es una hoja)
    def __init__(self, label):
        self.label = label

    def predict(self, X):
        #Devuelve siempre el label, sin importar que valor tenga X, porque en una hoja no se toman más decisiones
        return self.label

class Parent(Node):
    #representa un nodo del árbol, pero en este caso si se toman decisiones
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold # umbral

    def predict(self, X):
        #revisa el valor de X en la posición feature_index
        if X[self.feature_index]<self.threshold:
            return self.left_child.predict(X) #si es más pequeño que el umbralsigue por el hijo izquierdo
        else:
            return self.right_child.predict(X) #si es más grande o igual sigue por el hijo derecho

class Criterion(ABC):
    #clase abstracta que define un criterio para medir que tan bueno es un split
    @abstractmethod
    def method(self, dataset):
        pass

class Gini(Criterion):
    #Criterio que calcula el índice de Gini de un dataset
    def method(self, dataset):
        #G(D)=1-sum(p_c^2)
        C=len(np.unique(dataset.y))
        gini=1
        for c in range(C):
            p_c=np.sum(dataset.y==c)/dataset.get_num_samples
            gini -= (p_c)**2
        return gini 

class Entropy(Criterion):
    #criterio que calcula la entropia de un dataset
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
    num_trees = 80     # number of decision trees
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

    # Nuevo input para Extra-Trees
    extra_trees = input("Use Extra-Trees optimization? (yes/no): ").lower().strip() == 'yes'

    mode=input(str("Mode (sequencial/parallel): ")).lower()
    while mode not in ['sequencial', 'parallel']:        
        mode = input("Invalid mode. Choose (sequencial/parallel): ").lower()
    print("\n")

    time_start=time.time()
    rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterio, extra_trees)
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

    print('\n\nAccuracy {} %\n'.format(100*np.round(accuracy,decimals=2)))
    logging.info('Accuracy: %s', 100*np.round(accuracy,decimals=2))
    time_end = time.time()  # <-- Detener temporizador TOTAL
    total_time = time_end-time_start
    
    print(f'Total Time: {int(total_time // 60)}min {(total_time % 60):.4f}s\n')  # Formato MM min SS sec
    logging.info('Total Time: %d min %.4fs\n', int(total_time // 60), (total_time % 60))

    logging.info('----- Script ended -----')