from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DummyScaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return X

def get_scaler(scaler_type):
    if scaler_type=='standard':
        return StandardScaler()
    elif scaler_type=='minmax':
        return MinMaxScaler()
    else:
        return DummyScaler()