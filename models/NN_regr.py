import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

class NN_regressor:

    def __init__(self):
        self.model_name = 'NN'
        pass

    def fit(self, X, y):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        assert len(X.shape) == 2
        N, d = X.shape

        from keras.models import Sequential
        from keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers import SGD
        model = Sequential()
        model.add(Dense(10, input_dim=d, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1, activation="relu"))
        model.compile(loss="mse", optimizer=SGD(learning_rate=0.1))
        self.model = model

        n_epochs = 100
        self.model.fit(X, y, epochs=n_epochs, verbose=False)

    def predict(self, X):
        
        if type(X) == list:
            X = np.array(X)
            X = np.row_stack((X,X))
            return self.model.predict(X)[0]
        else:
            return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)

    def r2(self, y, pred):
        return r2_score(y,pred)

    def pearson(self, y, pred):
        return pearsonr(y,pred)

    