from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from nclick.utils.log import Log
from nclick.model.cv import GetCV
from nclick.model.gbdt import ModelGBDT, plot_feature_importance, xgb_logger, xgb_eval_multi_accuracy
from nclick.model.util import plot_metric_transition
from nclick.utils.io import IO


if __name__  == '__main__':

    # ==============================
    # データ準備
    # ==============================
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=0)
    Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    num_cols = [f'num_{i}' for i in range(10)]
    cat_cols = [f'cat_{i}' for i in range(10)]
    Xy = pd.DataFrame(Xy, columns=num_cols+cat_cols+['target'])
    Xy[cat_cols] = (Xy[cat_cols]//0.2).astype(str)
    Xy_train, Xy_test = train_test_split(Xy, test_size=0.2, shuffle=True, random_state=0)

    # ==============================
    # 設定
    # ==============================
    output_dir = Path('./result_xgb')
    logger_name = 'xgb'
    log_path = output_dir/f'{logger_name}.log'

    n_splits = 3
    n_repeats = 2
    target_col = ['target']
    important_features = ['num_0', 'num_1']
    exclude_features = []
    categorical_features = cat_cols

    gbdt_params = {
        'model_type': 'xgb',
        'objective':'multi:softprob',
        'num_class': 10,
        'learning_rate': 0.05,
        'max_depth': 5,
        'gamma':0.0,
        'colsample_bytree': 0.75,
        'subsample': 0.9,
        'feval': xgb_eval_multi_accuracy,
        'num_round': 10000,
        'early_stopping_rounds':50,
        'verbose': False,
        'maximize': True,
        'callbacks': [xgb_logger(logger_name, verbose=10)],
        'seed': 0,
        }


    gbdt_config = {
        'model_type': gbdt_params.pop('model_type'),
        'params': gbdt_params,
        'important_features': important_features,
        'exclude_features': exclude_features,
        'target': target_col,
        'sampling_feature_rate': 1.0,
        'imputation_method': {'numerical': [None, 'mean', 'median', -1e6][0],
                              'categorical': [None, 'most_frequent', -1e6][0]},
        'categorical_features': categorical_features,
        'categorical_encoder_type': ['label', 'astype_category', 'target'][0],
        'logger_name': logger_name,
    }

    # ==============================
    # 学習・推論
    # ==============================
    Log.create_logger(logger_name, log_path)
    Log.emphasis(f'Configure', logger_name, 'info', 50)
    Log.write(f'n_repeats: {n_repeats}, n_splits: {n_splits}', logger_name, 'info')
    Log.write(f'categorical_features: {categorical_features}',logger_name, 'info')
    Log.write( f'gbdt_params: \n{gbdt_params}', logger_name, 'info')

    y_pred_valid = np.zeros((Xy_train.shape[0], 10))
    y_pred_test = np.zeros((Xy_test.shape[0], 10))
    feature_importance = pd.DataFrame([])
    kfold = GetCV.KFold(Xy_train, n_repeats, n_splits, seed=0)
    for fold_idx, (trn_idx, val_idx) in enumerate(kfold):
        Log.emphasis(f'Fold{fold_idx}: {time.ctime()}', logger_name, 'info', 50)

        model_name = f"{gbdt_config['model_type']}_fold{fold_idx}"
        gbdt_config['model_name'] = model_name

        Xy_trn = Xy_train.iloc[trn_idx]
        Xy_val = Xy_train.iloc[val_idx]

        model = ModelGBDT(gbdt_config)
        model.train(Xy_trn, Xy_trn[target_col], Xy_val, Xy_val[target_col])
        IO.dump_pickle(model, output_dir/f'{model_name}.pickle')

        y_pred_val_fold = model.predict(Xy_val)
        y_pred_test_fold = model.predict(Xy_test)

        y_pred_valid[val_idx] += y_pred_val_fold / n_repeats
        y_pred_test += y_pred_test_fold / (n_splits*n_repeats)

        feature_importance = pd.concat([feature_importance, model.get_feature_importance()])
        plot_metric_transition(model)


    feature_importance = (feature_importance.groupby('feature')
                                            .gain.mean()
                                            .reset_index()
                                            .sort_values('gain', ascending=False)
                                            .reset_index(drop=True))
    plot_feature_importance(feature_importance['feature'], feature_importance['gain'], N=30)