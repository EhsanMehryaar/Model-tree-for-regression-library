from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
class gp_regr:

    def __init__(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        self.kernel = DotProduct() + WhiteKernel()
        self.model = GaussianProcessRegressor(kernel=self.kernel,random_state=0)

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