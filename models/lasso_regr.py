from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

class lasso_regr:

    def __init__(self):

        from sklearn.linear_model import Lasso
        self.model = Lasso()
        self.model_name = 'lasso'

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)
    
    def r2(self, y, pred):
        return r2_score(y,pred)

    def pearson(self, y, pred):
        return pearsonr(y,pred)[0]