class RandomForestClassifier:
    def __init__(self,num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion):
        self.num_trees = num_trees
        self.min_size = min_size
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.num_random_features = num_random_features
        self.criterion = criterion
        self.trees = []