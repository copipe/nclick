from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nclick.model.categorical_encoder import get_categorical_encoder
from nclick.model.imputer import get_imputer
from nclick.model.scaler import get_scaler


class Model(metaclass=ABCMeta):

    def __init__(self,
                 model_type: str,
                 model_name: str,
                 params: dict,
                 important_features: list = [],
                 exclude_features: list = [],
                 target: list = [],
                 sampling_feature_rate: float = 1.0,
                 imputation_method: dict = {'numerical': None, 'categorical': None},
                 categorical_features: list = [],
                 categorical_encoder_type: str = 'label',
                 scaler_type: str = 'standard',
                 logger_name: str = None,
                ) -> None:
        self.model_type = model_type
        self.model_name = model_name
        self.params = params
        self.model = None
        self.evals_result = {}
        self.important_features = important_features
        self.exclude_features = exclude_features
        self.target = target
        self.sampling_feature_rate = sampling_feature_rate
        self.features = None
        self.numerical_features = None
        self.categorical_features = categorical_features
        self.imputation_method = imputation_method
        self.imputers = {'numerical': None, 'categorical': None}
        self.categorical_encoder_type = categorical_encoder_type
        self.categorical_encoders = {}
        self.scaler_type = scaler_type
        self.logger_name = logger_name

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
                va_x: Optional[pd.DataFrame] = None,
                va_y: Optional[pd.Series] = None) -> None:
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass

    def _sampling_feature(self, tr_x, va_x):
        columns = tr_x.columns
        role_determined_features = self.important_features + self.exclude_features + self.target
        sampling_feature_candidate = np.array([f for f in columns if f not in role_determined_features])
        sampled_feature_indices = np.random.random(len(sampling_feature_candidate)) <= self.sampling_feature_rate
        sampled_feature = list(sampling_feature_candidate[sampled_feature_indices])
        self.features = self.important_features + sampled_feature
        self.categorical_features = [f for f in self.features if f in self.categorical_features]
        self.numerical_features = [f for f in self.features if f not in self.categorical_features]
        return tr_x[self.features], va_x[self.features]

    def _impute_missing_values(self, tr_x, va_x):
        num_imputer = get_imputer(self.imputation_method['numerical'])
        cat_imputer = get_imputer(self.imputation_method['categorical'])
        num_imputer.fit(tr_x[self.numerical_features])
        cat_imputer.fit(tr_x[self.categorical_features])
        tr_x[self.numerical_features] = num_imputer.transform(tr_x[self.numerical_features])
        tr_x[self.categorical_features] = cat_imputer.transform(tr_x[self.categorical_features])
        self.imputers['numerical'] = num_imputer
        self.imputers['categorical'] = cat_imputer
        return tr_x, va_x

    def _categorical_encoding(self, tr_x, tr_y, va_x, va_y):
        for feature in self.categorical_features:
            encoder = get_categorical_encoder(self.categorical_encoder_type)
            encoder.fit(tr_x, tr_y, feature)
            tr_x[feature] = encoder.transform(tr_x)
            va_x[feature] = encoder.transform(va_x)
            self.categorical_encoders[feature] = encoder
        return tr_x, va_x

    def _scale_numerical_values(self, tr_x, va_x):
        self.scaler = get_scaler(self.scaler_type)
        self.scaler.fit(tr_x[self.numerical_features])
        tr_x[self.numerical_features] = self.scaler.transform(tr_x[self.numerical_features])
        va_x[self.numerical_features] = self.scaler.transform(va_x[self.numerical_features])
        return tr_x, va_x

def plot_metric_transition(model):
    evals_result = model.evals_result.copy()
    best_iteration = evals_result.pop('best_iteration')
    datanames = list(evals_result.keys())
    eval_metrics = list(evals_result[datanames[0]].keys())

    for eval_metric in eval_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(eval_metric, fontsize=14)
        for dataname in datanames:
            y = model.evals_result[dataname][eval_metric]
            x = np.arange(len(y))
            ax.plot(x, y, label=dataname)
        ax.axvline(best_iteration, color='gray', linestyle='--', label=f'best iteration({best_iteration})')
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        plt.legend()
        plt.show()

def softmax(a):
    c = np.max(a, axis=1).reshape((-1, 1))
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1).reshape((-1, 1))
    y = exp_a / sum_exp_a
    return y

def clfpred2regpred(y_pred, classes, soft=True, softmax_func=False):
    if softmax_func:
        y_pred = softmax(y_pred)
    pred_soft = np.sum(y_pred * classes, axis=1)
    pred_hard = np.argmax(y_pred, axis=1)[classes]
    if soft:
        return pred_soft
    else:
        return pred_hard