
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
import xgboost as xgb
from catboost import CatBoost, Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import accuracy_score

from nclick.utils.log import Log
from nclick.model.util import Model


class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x, va_y):

        tr_x, va_x = self._sampling_feature(tr_x, va_x)
        tr_x, va_x = self._impute_missing_values(tr_x, va_x)
        tr_x, va_x = self._categorical_encoding(tr_x, tr_y, va_x, va_y)
        dtrain = lgb.Dataset(tr_x, tr_y)
        dvalid = lgb.Dataset(va_x, va_y)
        if self.logger_name:
            Log.write(f'(tr_x.shape, tr_y.shape): {tr_x.shape, tr_y.shape}', self.logger_name, 'info')
            Log.write(f'(va_x.shape, va_y.shape): {va_x.shape, va_y.shape}', self.logger_name, 'info')

        params = dict(self.params)
        num_round = params.pop('num_round')
        feval = params.pop('feval')
        callbacks = params.pop('callbacks')
        early_stopping_rounds = params.pop('early_stopping_rounds')

        self.model = lgb.train(
                            params,
                            dtrain,
                            num_boost_round=num_round,
                            valid_sets=(dtrain, dvalid),
                            feval=feval,
                            evals_result=self.evals_result,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=False,
                            callbacks=callbacks,
                            )
        self.evals_result['best_iteration'] = self.model.best_iteration

    def predict(self, te_x):
        te_x = te_x[self.features]
        te_x[self.numerical_features] = self.imputers['numerical'].transform(te_x[self.numerical_features])
        te_x[self.categorical_features] = self.imputers['categorical'].transform(te_x[self.categorical_features])
        for feature in self.categorical_features:
            te_x[feature] = self.categorical_encoders[feature].transform(te_x)
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def get_feature_importance(self):
        return pd.DataFrame({
                   'model_name': self.model_name,
                   'feature': self.features,
                   'gain': self.model.feature_importance('gain'),
               })

class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x, va_y):

        tr_x, va_x = self._sampling_feature(tr_x, va_x)
        tr_x, va_x = self._impute_missing_values(tr_x, va_x)
        tr_x, va_x = self._categorical_encoding(tr_x, tr_y, va_x, va_y)
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        if self.logger_name:
            Log.write(self.logger_name, 'info', f'(tr_x.shape, tr_y.shape): {tr_x.shape, tr_y.shape}')
            Log.write(self.logger_name, 'info', f'(va_x.shape, va_y.shape): {va_x.shape, va_y.shape}')

        # ハイパーパラメータの設定
        params = self.params.copy()
        num_round = params.pop('num_round')
        verbose = params.pop('verbose')
        feval = params.pop('feval')
        maximize = params.pop('maximize')
        early_stopping_rounds = params.pop('early_stopping_rounds')
        callbacks = params.pop('callbacks')

        # 学習
        self.model = xgb.train(
                                params,
                                dtrain,
                                num_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=verbose,
                                evals_result=self.evals_result,
                                feval=feval,
                                maximize=maximize,
                                callbacks=callbacks,
        )
        self.evals_result['best_iteration'] = self.model.best_iteration

    def predict(self, te_x):
        te_x = te_x[self.features]
        te_x[self.numerical_features] = self.imputers['numerical'].transform(te_x[self.numerical_features])
        te_x[self.categorical_features] = self.imputers['categorical'].transform(te_x[self.categorical_features])
        for feature in self.categorical_features:
            te_x[feature] = self.categorical_encoders[feature].transform(te_x)
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def get_feature_importance(self):
        fe = [[self.model_name, f, i] for f, i in self.model.get_score(importance_type='total_gain').items()]
        return pd.DataFrame(fe, columns=['model_name', 'feature', 'gain'])

class ModelCAT(Model):

    def train(self, tr_x, tr_y, va_x, va_y):

        tr_x, va_x = self._sampling_feature(tr_x, va_x)
        tr_x, va_x = self._impute_missing_values(tr_x, va_x)
        tr_x, va_x = self._categorical_encoding(tr_x, tr_y, va_x, va_y)
        if self.categorical_encoder_type == 'astype_category':
            dtrain = Pool(tr_x, label=tr_y, cat_features=self.categorical_features)
            dvalid = Pool(va_x, label=va_y, cat_features=self.categorical_features)
        else:
            dtrain = Pool(tr_x, label=tr_y)
            dvalid = Pool(va_x, label=va_y)
        eval_set = [dvalid]
        if self.logger_name:
            Log.write(self.logger_name, 'info', f'(tr_x.shape, tr_y.shape): {tr_x.shape, tr_y.shape}')
            Log.write(self.logger_name, 'info', f'(va_x.shape, va_y.shape): {va_x.shape, va_y.shape}')

        params = self.params.copy()
        early_stopping_rounds = params.pop('early_stopping_rounds')
        self.task_is_regression = params.pop('task_is_regression')

        # 学習
        if self.task_is_regression:
            self.model = CatBoostRegressor(**params)
        else:
            self.model = CatBoostClassifier(**params)
        self.model.fit(dtrain,
                       eval_set=eval_set,
                       early_stopping_rounds=early_stopping_rounds,
                       use_best_model=True,
                      )
        self.evals_result = self.model.get_evals_result()
        self.evals_result['best_iteration'] = self.model.best_iteration_

    def predict(self, te_x):
        te_x = te_x[self.features]
        te_x[self.numerical_features] = self.imputers['numerical'].transform(te_x[self.numerical_features])
        te_x[self.categorical_features] = self.imputers['categorical'].transform(te_x[self.categorical_features])
        for feature in self.categorical_features:
            te_x[feature] = self.categorical_encoders[feature].transform(te_x)
        if self.task_is_regression:
            return self.model.predict(te_x)
        else:
            return self.model.predict_proba(te_x)

    def get_feature_importance(self):
        return pd.DataFrame({
                   'model_name': self.model_name,
                   'feature': self.model.feature_names_,
                   'gain': self.model.feature_importances_,
               })

def ModelGBDT(gbdt_config):
    model_type = gbdt_config['model_type']
    if model_type == 'lgb':
        return ModelLGB(**gbdt_config)
    if model_type == 'xgb':
        return ModelXGB(**gbdt_config)
    if model_type == 'cat':
        return ModelCAT(**gbdt_config)

def lgb_logger(logger_name, level='info', verbose=50, show_stdv=True):

    def _callback(env):
        if verbose > 0 and env.evaluation_result_list and (env.iteration + 1) % verbose == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            Log.write( "[%d]\t%s" % (env.iteration + 1, result), logger_name, level)
    _callback.order = 10
    return _callback

def xgb_logger(logger_name, level='info', verbose=50, show_stdv=True):

    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def _callback(env):
        if verbose > 0 and env.evaluation_result_list and (env.iteration + 1) % verbose == 0:
            result = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            Log.write("[%d]\t%s" % (env.iteration + 1, result), logger_name, level)
    _callback.order = 10
    return _callback

def plot_feature_importance(feature, importance, N):

    feature = [f'{f}[{i}]' for i, f in enumerate(feature, start=1)]
    feat = feature[:N][::-1]
    impo = importance.iloc[:N].iloc[::-1]


    fig, ax = plt.subplots(figsize=(8, N/2.5))
    ax.set_title('feature importance (gain)', fontsize=14)

    ax.barh(feat, impo)
    for topN in range(N//10):
        ax.axhline(topN*10, color='gray', linestyle='--')

    ax.set_axisbelow(True)
    ax.grid(axis='x')
    plt.show()

def lgb_eval_multi_accuracy(y_pred, data):
    y_true = data.get_label()
    y_pred = np.argmax(y_pred.reshape(len(y_true), -1), axis=1)
    acc = accuracy_score(y_true, y_pred)
    return 'multi_accuracy', acc , True

def xgb_eval_multi_accuracy(y_pred, data):
    y_true = data.get_label()
    y_pred = np.argmax(y_pred.reshape(len(y_true), -1), axis=1)
    acc = accuracy_score(y_true, y_pred)
    return 'multi_accuracy', acc 
