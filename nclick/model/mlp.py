import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nclick.utils.log import Log
from nclick.model.util import Model, clfpred2regpred


class TabularDataset(Dataset):
    def __init__(self, X, y, categorical_features=[], numerical_features=[]):

        self.n_samples = X.shape[0]
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        if y is None:
            self.y = np.full(self.n_samples, np.nan, dtype=np.float32).reshape(-1, 1)
        else:
            self.y = y.astype(np.float32).values.reshape(-1, 1)

        if self.numerical_features:
            self.X_num = X[self.numerical_features].astype(np.float32).values
        else:
            self.X_num = np.zeros((self.n_samples, 1))

        if self.categorical_features:
            self.X_cat = X[self.categorical_features].astype(np.int64).values
        else:
            self.X_cat = np.zeros((self.n_samples, 1))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [self.y[idx], self.X_num[idx], self.X_cat[idx]]

class MLP(nn.Module):
    def __init__(
        self,
        embedding_dims,
        n_numerical_features,
        linear_layer_sizes,
        output_size,
        embedding_dropout,
        linear_layer_dropouts,
    ):

        super().__init__()

        # Embedding layers
        self.embedding_layers = nn.ModuleList([nn.Embedding(i, o) for i, o in embedding_dims])
        n_embedded_features = sum([o for i, o in embedding_dims])
        self.n_embedded_features = n_embedded_features

        # Linear Layers
        self.n_numerical_features = n_numerical_features
        first_linear_layer = nn.Linear(self.n_embedded_features + self.n_numerical_features, linear_layer_sizes[0])
        self.linear_layers = nn.ModuleList(
            [first_linear_layer] +
            [nn.Linear(linear_layer_sizes[i], linear_layer_sizes[i + 1]) for i in range(len(linear_layer_sizes) - 1)]
        )

        # Output Layer
        self.output_layer = nn.Linear(linear_layer_sizes[-1], output_size)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.n_numerical_features)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in linear_layer_sizes])

        # Dropout Layers
        self.embedding_dropout_layer = nn.Dropout(embedding_dropout)
        self.linear_droput_layers = nn.ModuleList([nn.Dropout(size) for size in linear_layer_dropouts])

        # Initialize Weight
        for linear_layer in self.linear_layers:
            nn.init.kaiming_normal_(linear_layer.weight.data)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

    def forward(self, X_num, X_cat):

        if self.n_embedded_features != 0:
            x = [
                embedding_layer(X_cat[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)
            x = self.embedding_dropout_layer(x)

        if self.n_numerical_features != 0:
            normalized_X_num = self.first_bn_layer(X_num)

            if self.n_embedded_features != 0:
                x = torch.cat([x, normalized_X_num], 1)
            else:
                x = normalized_X_num

        for linear_layer, dropout_layer, bn_layer in zip(
            self.linear_layers, self.linear_droput_layers, self.bn_layers
        ):

            x = F.relu(linear_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x

class ModelMLP(Model):
    def train(self, tr_x, tr_y, va_x, va_y):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tr_x, va_x = self._sampling_feature(tr_x, va_x)
        tr_x, va_x = self._impute_missing_values(tr_x, va_x)
        tr_x, va_x = self._categorical_encoding(tr_x, tr_y, va_x, va_y)
        tr_x, va_x = self._scale_numerical_values(tr_x, va_x)
        if self.logger_name:
            Log.write(self.logger_name, 'info', f'(tr_x.shape, tr_y.shape): {tr_x.shape, tr_y.shape}')
            Log.write(self.logger_name, 'info', f'(va_x.shape, va_y.shape): {va_x.shape, va_y.shape}')

        params = self.params.copy()
        n_epochs = params.pop('n_epochs')
        batchsize = params.pop('batchsize')
        max_embedding_dim = params.pop('max_embedding_dim')
        dataset = params.pop('Dataset')
        dataloader = params.pop('DataLoader')
        criterion = params.pop('criterion')
        optimizer = params.pop('optimizer')
        weight_decay = params.pop('weight_decay')
        lr = params.pop('lr')
        verbose = params.pop('verbose')

        ds_train = dataset(X=tr_x,
                           y=tr_y,
                           categorical_features=self.categorical_features,
                           numerical_features=self.numerical_features)
        ds_valid = dataset(X=va_x,
                           y=va_y,
                           categorical_features=self.categorical_features,
                           numerical_features=self.numerical_features)
        dl_train= dataloader(ds_train, batchsize, shuffle=True)
        dl_valid= dataloader(ds_valid, batchsize, shuffle=False)

        categorical_dims = [int(tr_x[c].nunique()) for c in self.categorical_features]
        embedding_dims = [(x, min(max_embedding_dim, (x + 1) // 2)) for x in categorical_dims]
        self.model = MLP(embedding_dims,
                         n_numerical_features=len(self.numerical_features),
                         **params,
                         ).to(self.device)
        self.criterion = criterion()
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        self.evals_result['train'] = {}
        self.evals_result['valid'] = {}
        self.evals_result['train']['loss'] = []
        self.evals_result['valid']['loss'] = []
        best_valid_loss = np.inf
        best_state_dict = None
        for epoch in range(n_epochs):

            self.model.train()
            avg_train_loss = 0
            y_true_trns = np.zeros(len(tr_x))
            y_pred_trns = np.zeros(len(tr_x))
            for i, (y, X_num, X_cat) in enumerate(dl_train):
                X_cat = X_cat.to(self.device)
                X_num = X_num.to(self.device)
                y = y.to(self.device)

                # Forward Pass
                preds = self.model(X_num, X_cat)
                loss = self.criterion(preds, y)

                # Backward Pass and Optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_train_loss += loss.item()*len(y) / len(ds_train)
                y_true_trns[i*batchsize:(i+1)*batchsize] = y.detach().cpu().numpy().ravel()
                y_pred_trns[i*batchsize:(i+1)*batchsize] = preds.detach().cpu().numpy().ravel()

            self.model.eval()
            avg_valid_loss = 0
            y_true_vals = np.zeros(len(va_x))
            y_pred_vals = np.zeros(len(va_x))
            with torch.no_grad():
                for i, (y, X_num, X_cat) in enumerate(dl_valid):
                    X_cat = X_cat.to(self.device)
                    X_num = X_num.to(self.device)
                    y = y.to(self.device)

                    # Forward Pass
                    preds = self.model(X_num, X_cat)
                    loss = self.criterion(preds, y)
                    avg_valid_loss += loss.item()*len(y) / len(ds_valid)
                    y_true_vals[i*batchsize:(i+1)*batchsize] = y.detach().cpu().numpy().ravel()
                    y_pred_vals[i*batchsize:(i+1)*batchsize] = preds.detach().cpu().numpy().ravel()

                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    best_state_dict = self.model.state_dict

                scheduler.step(avg_valid_loss)
                if epoch % verbose == 0:
                    log_msg = f'[{epoch}]    train-loss:{avg_train_loss:.4f}    valid-loss:{avg_valid_loss:.4f}' 
                    if self.logger_name:
                        Log.write(self.logger_name, 'info', log_msg)
                    else:
                        print(log_msg)

            self.evals_result['train']['loss'].append(avg_train_loss)
            self.evals_result['valid']['loss'].append(avg_valid_loss)

        self.evals_result['best_iteration'] = np.argmin(self.evals_result['valid']['loss'])
        self.model.load_state_dict(best_state_dict)

    def predict(self, te_x):
        te_x = te_x[self.features]
        te_x[self.numerical_features] = self.imputers['numerical'].transform(te_x[self.numerical_features])
        te_x[self.categorical_features] = self.imputers['categorical'].transform(te_x[self.categorical_features])
        for feature in self.categorical_features:
            te_x[feature] = self.categorical_encoders[feature].transform(te_x)
        te_x[self.numerical_features] = self.scaler.transform(te_x[self.numerical_features])

        batchsize = self.params['batchsize']
        dataset = self.params['Dataset']
        dataloader = self.params['DataLoader']
        ds_test = dataset(X=te_x,
                          y=None,
                          categorical_features=self.categorical_features,
                          numerical_features=self.numerical_features)
        dl_test = dataloader(ds_test, batchsize, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            preds = np.zeros(len(te_x))
            for i, (y, X_num, X_cat) in enumerate(dl_test):
                X_cat = X_cat.to(self.device)
                X_num = X_num.to(self.device)
                preds[i*batchsize:(i+1)*batchsize] = self.model(X_num, X_cat).detach().cpu().numpy().ravel()
        return preds

    def get_feature_importance(self):
        pass

class ModelMLP_CLFREG(Model):
    def train(self, tr_x, tr_y, va_x, va_y, classes, weight):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tr_x, va_x = self._sampling_feature(tr_x, va_x)
        tr_x, va_x = self._impute_missing_values(tr_x, va_x)
        tr_x, va_x = self._categorical_encoding(tr_x, tr_y, va_x, va_y)
        tr_x, va_x = self._scale_numerical_values(tr_x, va_x)
        if self.logger_name:
            Log.write(self.logger_name, 'info', f'(tr_x.shape, tr_y.shape): {tr_x.shape, tr_y.shape}')
            Log.write(self.logger_name, 'info', f'(va_x.shape, va_y.shape): {va_x.shape, va_y.shape}')

        params = self.params.copy()
        self.classes = classes
        n_epochs = params.pop('n_epochs')
        batchsize = params.pop('batchsize')
        max_embedding_dim = params.pop('max_embedding_dim')
        dataset = params.pop('Dataset')
        dataloader = params.pop('DataLoader')
        criterion = params.pop('criterion')
        optimizer = params.pop('optimizer')
        weight_decay = params.pop('weight_decay')
        lr = params.pop('lr')
        verbose = params.pop('verbose')

        ds_train = dataset(X=tr_x,
                           y=tr_y,
                           categorical_features=self.categorical_features,
                           numerical_features=self.numerical_features)
        ds_valid = dataset(X=va_x,
                           y=va_y,
                           categorical_features=self.categorical_features,
                           numerical_features=self.numerical_features)
        dl_train= dataloader(ds_train, batchsize, shuffle=True, num_workers=8)
        dl_valid= dataloader(ds_valid, batchsize, shuffle=False, num_workers=8)

        categorical_dims = [int(tr_x[c].nunique()) for c in self.categorical_features]
        embedding_dims = [(x, min(max_embedding_dim, (x + 1) // 2)) for x in categorical_dims]
        self.model = MLP(embedding_dims,
                         n_numerical_features=len(self.numerical_features),
                         **params,
                         ).to(self.device)
        self.criterion = criterion(weight=torch.tensor(weight).to(self.device).float())
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        self.evals_result['train'] = {}
        self.evals_result['valid'] = {}
        self.evals_result['train']['loss'] = []
        self.evals_result['valid']['loss'] = []
        best_valid_loss = np.inf
        best_state_dict = None
        for epoch in range(n_epochs):

            self.model.train()
            avg_train_loss = 0
            y_true_trns = np.zeros(len(tr_x))
            y_pred_trns = np.zeros(len(tr_x))
            for i, (y, X_num, X_cat) in enumerate(dl_train):
                X_cat = X_cat.to(self.device)
                X_num = X_num.to(self.device)
                y = y.to(self.device).long().reshape(-1)

                # Forward Pass
                preds = self.model(X_num, X_cat)
                loss = self.criterion(preds, y)

                # Backward Pass and Optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_train_loss += loss.item()*len(y) / len(ds_train)
                y_true_trns[i*batchsize:(i+1)*batchsize] = y.detach().cpu().numpy().ravel()
                y_pred_trns[i*batchsize:(i+1)*batchsize] = clfpred2regpred(preds.detach().cpu().numpy(), self.classes, soft=True, softmax_func=True)

            self.model.eval()
            avg_valid_loss = 0
            y_true_vals = np.zeros(len(va_x))
            y_pred_vals = np.zeros(len(va_x))
            with torch.no_grad():
                for i, (y, X_num, X_cat) in enumerate(dl_valid):
                    X_cat = X_cat.to(self.device)
                    X_num = X_num.to(self.device)
                    y = y.to(self.device).long().reshape(-1)

                    # Forward Pass
                    preds = self.model(X_num, X_cat)
                    loss = self.criterion(preds, y)
                    avg_valid_loss += loss.item()*len(y) / len(ds_valid)
                    y_true_vals[i*batchsize:(i+1)*batchsize] = y.detach().cpu().numpy().ravel()
                    y_pred_vals[i*batchsize:(i+1)*batchsize] = clfpred2regpred(preds.detach().cpu().numpy(), self.classes, soft=True, softmax_func=True)

                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    best_state_dict = self.model.state_dict()

                scheduler.step(avg_valid_loss)
                if epoch % verbose == 0 and epoch != 0:
                    log_msg = f'[{epoch}]    train-loss:{avg_train_loss:.4f}    valid-loss:{avg_valid_loss:.4f}' 
                    if self.logger_name:
                        Log.write(self.logger_name, 'info', log_msg)
                    else:
                        print(log_msg)

            self.evals_result['train']['loss'].append(avg_train_loss)
            self.evals_result['valid']['loss'].append(avg_valid_loss)

        self.evals_result['best_iteration'] = np.argmin(self.evals_result['valid']['loss'])
        self.model.load_state_dict(best_state_dict)

    def predict(self, te_x):
        te_x = te_x[self.features]
        te_x[self.numerical_features] = self.imputers['numerical'].transform(te_x[self.numerical_features])
        te_x[self.categorical_features] = self.imputers['categorical'].transform(te_x[self.categorical_features])
        for feature in self.categorical_features:
            te_x[feature] = self.categorical_encoders[feature].transform(te_x)
        te_x[self.numerical_features] = self.scaler.transform(te_x[self.numerical_features])

        batchsize = self.params['batchsize']
        dataset = self.params['Dataset']
        dataloader = self.params['DataLoader']
        ds_test = dataset(X=te_x,
                          y=None,
                          categorical_features=self.categorical_features,
                          numerical_features=self.numerical_features)
        dl_test = dataloader(ds_test, batchsize, shuffle=False, num_workers=8)
        self.model.eval()
        with torch.no_grad():
            preds = np.zeros(len(te_x))
            for i, (y, X_num, X_cat) in enumerate(dl_test):
                X_cat = X_cat.to(self.device)
                X_num = X_num.to(self.device)
                preds[i*batchsize:(i+1)*batchsize] = clfpred2regpred(self.model(X_num, X_cat).detach().cpu().numpy(), self.classes, soft=True, softmax_func=True)
        return preds

    def get_feature_importance(self):
        pass