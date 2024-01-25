import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, importance=0.85, estimator=None, beta=None):
        self.importance = importance
        self.estimator = estimator
        self.feature_indices_ = None
        self.beta = beta
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Perform SHAP analysis
        explainer = shap.TreeExplainer(self.estimator)
        shap_values = explainer.shap_values(X)

        if not self.beta:
            shap_summaries = np.abs(shap_values[1]).mean(axis=0)
            self.feature_indices_ = np.argsort(shap_summaries)[-int(self.importance*X.shape[1]):]
        else:
            shap_cm_df = self.create_shap_cm_df(X, y, shap_values)
            shap_cm_df['recall'], shap_cm_df['precision'], shap_cm_df['f_beta'] = \
                (zip(*shap_cm_df.apply(lambda row: self.get_f_beta_score(
                    row['TP'], row['FP'], row['TN'], row['FN'], beta=self.beta), axis=1)))
            self.feature_names_ = list(shap_cm_df[shap_cm_df.f_beta > 1-self.importance].index)
        return self

    def transform(self, X):
        if not self.beta:
            return X.iloc[:, self.feature_indices_]
        return X[self.feature_names_]

    @staticmethod
    def get_f_beta_score(TP, FP, TN, FN, beta=0.5):
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f_beta_score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2) * precision + recall) if precision * recall != 0 else 0
        return recall, precision, f_beta_score

    @staticmethod
    def create_shap_cm_df(x, y=None, shap_values=None):
        positive_mask = np.array(y) > 0
        negative_mask = np.array(y) < 1
        shap_df = pd.DataFrame(shap_values[1], columns=list(x.columns))
        TP_ser = (shap_df[positive_mask].clip(lower=0) > 0).sum()
        FP_ser = (shap_df[negative_mask].clip(lower=0) > 0).sum()
        TN_ser = (shap_df[negative_mask].clip(upper=0) < 0).sum()
        FN_ser = (shap_df[positive_mask].clip(upper=0) < 0).sum()
        cf_list = [TP_ser, FP_ser, TN_ser, FN_ser]
        cf_df = pd.concat(cf_list, axis=1)
        cf_df.columns = ['TP', 'FP', 'TN', 'FN']
        return cf_df
