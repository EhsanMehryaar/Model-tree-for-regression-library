import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
class mean_regr:

    def __init__(self):
        self.y_mean = None

    def fit(self, X, y):
        self.y_mean = np.sum(y) / len(y)

    def predict(self, X):
        assert self.y_mean is not None
        return self.y_mean * np.ones(len(X))

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)

    def pearson(self, y, pred):
        return pearsonr(y,pred)
