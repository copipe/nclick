import numpy as np
from sklearn.impute import SimpleImputer


def get_imputer(strategy):
    if isinstance(strategy, int) or isinstance(strategy, float):
        return SimpleImputer(strategy='constant', fill_value=strategy)
    elif isinstance(strategy, str):
        return SimpleImputer(strategy=strategy)
    else:
        return SimpleImputer(strategy='constant', fill_value=np.nan)