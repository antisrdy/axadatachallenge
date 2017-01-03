from sklearn.base import BaseEstimator
from xgboost import sklearn
import numpy as np
from sklearn.linear_model import Lasso

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = sklearn.XGBRegressor(max_depth=3,
                            learning_rate=0.1,
                            n_estimators=200,
                            silent=True,
                            objective='reg:linear',
                            gamma=0,
                            min_child_weight=1,
                            max_delta_step=0,
                            subsample=1,
                            colsample_bytree=1,
                            colsample_bylevel=0.25,
                            reg_alpha=0 ,
                            reg_lambda=0.5,
                            scale_pos_weight=1,
                            base_score=0.5,
                            seed=0,
                            missing=None)
        #self.clf = Lasso()
        

    def fit(self, X, y):
        labels = y.reshape((-1,1))
        self.clf.fit(X, labels)

    def predict(self, X):
        predicted = self.clf.predict(X)
        return predicted