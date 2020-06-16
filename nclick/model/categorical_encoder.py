import numpy as np
import pandas as pd

class EncoderMixin:

    def __init__(self):
        self.encoder_dict = None
        self.feature = None
        self.classes = None
        self.default = None

    def _np_map(self, func, *iterables):
        return np.array(list(map(func, *iterables)))

    def _encoder(self, category):
        return self.encoder_dict.get(category, self.default)

    def _encode(self, x):
        return self._np_map(self._encoder, x)

    def transform(self, X):
        x = X[self.feature].values
        x_encoded = self._encode(x)
        return x_encoded

class LabelEncoder(EncoderMixin):

    def fit(self, X, y, feature):
        x = X[feature].values

        # confugure
        self.feature = feature
        self.classes, value_counts = np.unique(x, return_counts=True)
        self.classes = self.classes[np.argsort(value_counts)]
        self.default = np.argmax(self.classes)

        # encode
        self.encoder_dict = {cat: i for i, cat in enumerate(self.classes)}

class AsTypeCategory(EncoderMixin):

    def fit(self, X, y, feature):
        x = X[feature].values

        # confugure
        self.feature = feature
        self.classes = np.unique(x)

    def transform(self, X):
        return X[self.feature].astype('category')

class TargetEncoder(EncoderMixin):

    def fit(self, X, y, feature):
        avg_target_by_cat = (pd.concat([X[[feature]], y], axis=1)
                               .groupby(feature).mean())

        # confugure
        self.feature = feature
        self.classes = np.array(avg_target_by_cat.index)
        self.default = y.values.mean()

        # encode
        self.encoder_dict = dict(zip(avg_target_by_cat.index, avg_target_by_cat.values))

def get_categorical_encoder(categorical_encoder_type='label'):
    if categorical_encoder_type=='label':
        return LabelEncoder()
    elif categorical_encoder_type=='target':
        return TargetEncoder()
    elif categorical_encoder_type=='astype_category':
        return AsTypeCategory()