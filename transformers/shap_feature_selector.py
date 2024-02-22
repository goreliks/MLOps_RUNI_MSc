import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from inspect import isfunction
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, importance=0.85, estimator=None, beta=None):
        self.importance = importance
        self.estimator = estimator
        self.feature_indices_ = None
        self.beta = beta

    def fit(self, X, y=None):
        # Perform SHAP analysis
        if isinstance(self.estimator, lgb.basic.Booster):
            explainer = shap.TreeExplainer(self.estimator)
            shap_values = explainer.shap_values(X)
        elif isfunction(self.estimator):
            background_data = shap.sample(X, nsamples=100)
            explainer = shap.KernelExplainer(self.estimator, background_data)
            X['label'] = y
            # x_explain = shap.sample(X, nsamples=400)
            x_explain = X.iloc[:400]
            y = x_explain['label'].values
            X = X.drop(['label'], axis=1)
            x_explain = x_explain.drop(['label'], axis=1)
            shap_values = explainer.shap_values(x_explain, nsamples=100)
            shap_values = [shap_values*(-1), shap_values]
        else:
            raise ValueError("Unsupported model type for SHAPFeatureSelector")

        if not self.beta:
            shap_summaries = np.abs(shap_values[1]).mean(axis=0)
            self.feature_indices_ = np.argsort(shap_summaries)[-int(self.importance*X.shape[1]):]
        else:
            shap_cm_df = self.create_shap_cm_df(X, y, shap_values)
            shap_cm_df['recall'], shap_cm_df['precision'], shap_cm_df['f_beta'], shap_cm_df['contrib_coef'], \
                shap_cm_df['f_beta_score_normed'] = \
                (zip(*shap_cm_df.apply(lambda row: self.get_f_beta_score(
                    row['TP'], row['FP'], row['TN'], row['FN'], sample_num=len(y), beta=self.beta), axis=1)))
            mask = shap_cm_df['f_beta_score_normed'] > 1-self.importance
            if mask.sum() < int(shap_cm_df.shape[0]*0.7):
                shap_summaries = np.abs(shap_values[1]).mean(axis=0)
                self.feature_indices_ = np.argsort(shap_summaries)[-int(self.importance * X.shape[1]):]
                return self
            numeric_indexes = mask[mask].index.tolist()
            self.feature_indices_ = [shap_cm_df.index.get_loc(index) for index in numeric_indexes]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            return X[:, self.feature_indices_]

    @staticmethod
    def get_f_beta_score(TP, FP, TN, FN, sample_num, beta=2):
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f_beta_score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2) * precision + recall) if precision * recall != 0 else 0
        num_of_cont = TP + FP + TN + FN
        contr_coef = num_of_cont/sample_num
        f_beta_normed = contr_coef*f_beta_score
        return recall, precision, f_beta_score, contr_coef, f_beta_normed

    @staticmethod
    def create_shap_cm_df(x, y=None, shap_values=None):
        positive_mask = np.array(y) == 1
        negative_mask = np.array(y) == 0
        shap_df = pd.DataFrame(shap_values[1])
        TP_ser = (shap_df[positive_mask].clip(lower=0) > 0).sum()
        FP_ser = (shap_df[negative_mask].clip(lower=0) > 0).sum()
        TN_ser = (shap_df[negative_mask].clip(upper=0) < 0).sum()
        FN_ser = (shap_df[positive_mask].clip(upper=0) < 0).sum()
        cf_list = [TP_ser, FP_ser, TN_ser, FN_ser]
        cf_df = pd.concat(cf_list, axis=1)
        cf_df.columns = ['TP', 'FP', 'TN', 'FN']
        return cf_df
