import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class KaiserWindowSmoother(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=11, beta=2):
        self.window_length = window_length
        self.beta = beta

    def smooth_curve(self, x):
        s = np.r_[x[self.window_length-1:0:-1], x, x[-1:-self.window_length:-1]]
        w = np.kaiser(self.window_length, self.beta)
        y = np.convolve(w/w.sum(), s, mode='valid')
        return y[(self.window_length//2):-(self.window_length//2)]

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        X_smoothed = np.zeros_like(X.values)
        for i in range(X.shape[1]):
            X_smoothed[:, i] = self.smooth_curve(X.values[:, i].flatten())
        return X_smoothed
