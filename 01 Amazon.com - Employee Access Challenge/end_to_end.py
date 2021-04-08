import numpy as np
import pandas as pd
import gc
import os


# 클래스 연습 용

class TestMjPractice(object):
    def __init__(self, need):
        self.abc = None
        self.dez = '123'
        self.mj = 'MJ"'
        self.need = need



class TargetEncoding():
    def __init__(self, columns_names ):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan
        self.MJ = 'test'
        self.mj = self.__class__.MJ

    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            self.learned_values[c] = (X_[[c,"__target__"]]
                                      .groupby(c)["__target__"].mean()
                                      .reset_index())

        self.dataset_mean = np.mean(y)
        return self
    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on = c, how = 'left')
                               )["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        return transformed_X
    def fit_transform(self, X, y, **fit_params):
        self.fit(X,y)
        return self.transform(X)


from sklearn.base import BaseEstimator, TransformerMixin

te = TargetEncoding()

class TargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan

    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            self.learned_values[c] = (X_[[c, "__target__"]]
                                      .groupby(c)["__target__"].mean()
                                      .reset_index())
        self.dataset_mean = np.mean(y)
        return self

    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on=c, how='left')
                                )["__target__"]
            transformed_X = transformed_X.fillna(self.dataset_mean)
            return transformed_X

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def test():
        print('test')