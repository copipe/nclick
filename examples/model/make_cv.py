import numpy as np
import pandas as pd

from nclick.model.cv import GetCV

if __name__ == '__main__':
    data = pd.DataFrame([
        [0.0 ,0, 0],
        [0.1 ,0, 0],
        [0.2 ,0, 0],
        [0.3 ,1, 0],
        [0.4 ,1, 0],
        [0.5 ,1, 0],
        [0.6 ,2, 1],
        [0.7 ,2, 1],
        [0.8 ,2, 1],
        [0.9 ,3, 1],
        [1.0 ,3, 1],
        ], columns=['x', 'group', 'y'])

    kfold = GetCV.KFold(X=data, n_repeats=2, n_splits=3, seed=0)
    kfold = GetCV.StratifiedKFold(X=data, y=data['y'], n_repeats=2, n_splits=3, seed=0)
    kfold = GetCV.GroupKFold(X=data, groups=data['group'], n_repeats=2, n_splits=3, seed=0)
    kfold = GetCV.StratifiedGroupKFold(X=data, y=data['y'], groups=data['group'], n_repeats=2, n_splits=3, seed=0)