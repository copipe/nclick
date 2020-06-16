import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold, RepeatedKFold, RepeatedStratifiedKFold


class GetCV:
    @classmethod
    def KFold(cls, X, n_repeats, n_splits, seed=0):
        cv = []
        kfold = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=seed)
        for trn_idx, val_idx in kfold.split(X=X):
            cv.append([trn_idx, val_idx])
        return cv

    @classmethod
    def StratifiedKFold(cls, X, y, n_repeats, n_splits, seed=0):
        cv = []
        kfold = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=seed)
        for trn_idx, val_idx in kfold.split(X=X, y=y):
            cv.append([trn_idx, val_idx])
        return cv

    @classmethod
    def GroupKFold(cls, X, groups, n_repeats, n_splits, seed=0):
        np.random.seed(seed)
        cv = []
        idx_org = np.arange(len(X))
        for _ in range(n_repeats):
            idx = np.random.permutation(idx_org)
            kfold = GroupKFold(n_splits=n_splits)
            for trn_idx, val_idx in kfold.split(X.iloc[idx], groups=groups[idx]):
                cv.append([idx[trn_idx], idx[val_idx]])
        return cv

    @classmethod
    def _stratified_group_k_fold(cls, X, y, groups, k, seed=None):
        '''
        reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        '''
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

    @classmethod
    def StratifiedGroupKFold(cls, X, y, groups, n_repeats, n_splits, seed=0):
        np.random.seed(seed)
        cv = []
        idx_org = np.arange(len(X))
        for kfold_idx in range(n_repeats):
            idx = np.random.permutation(idx_org)
            kfold = GetCV._stratified_group_k_fold(X, y, groups, n_splits, seed=seed+kfold_idx)
            for trn_idx, val_idx in kfold:
                cv.append([idx[trn_idx], idx[val_idx]])
        return cv