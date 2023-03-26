from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
class rt_regr:

    def __init__(self, max_depth=20, min_samples_leaf=10):
        from sklearn.tree import DecisionTreeRegressor
        self.model = DecisionTreeRegressor(max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf,
                                           criterion="mse")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def loss(self, X, y):
        return mean_squared_error(y, self.predict(X))
    
    def r2(self, X, y):
        return r2_score(y, self.model.predict(X))

    def pearson(self, X, y):
        return pearsonr(y, self.model.predict(X))[0]