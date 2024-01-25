import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class TemporalFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, look_back=10):
        self.look_back = look_back
        self.features = ['A1_mean', 'A1_min', 'A1_max', 'A2_mean', 'A2_min', 'A2_max', 'Cur_mean', 'Cur_min', 'Cur_max',
                         'Pre_mean', 'Pre_min', 'Pre_max', 'Temp_mean', 'Temp_min', 'Temp_max', 'Ther_mean', 'Ther_min',
                         'Ther_max', 'Vol_mean', 'Vol_min', 'Vol_max', 'Flow_mean', 'Flow_min', 'Flow_max']

    def create_dataset(self, dataset):
        data_X = np.zeros((len(dataset) - self.look_back + 1, 3))
        j = 0
        for i in range(self.look_back - 1, len(dataset)):
            data_pre = dataset[i - self.look_back + 1:i + 1, 0]

            data_pre_mean = np.mean(data_pre, axis=0)
            data_pre_min = np.min(data_pre, axis=0)
            data_pre_max = np.max(data_pre, axis=0)

            data_X[j, :] = np.array([data_pre_mean, data_pre_min, data_pre_max])
            j += 1

        return np.array(data_X).reshape(-1, 3)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X is a 2D numpy array with shape (n_samples, n_features)
        for i in range(X.shape[1]):
            if i == 0:
                transformed_data = self.create_dataset(X[:, i].reshape(-1, 1))
            else:
                transformed_data = np.concatenate([transformed_data, self.create_dataset(X[:, i].reshape(-1, 1))],
                                                  axis=-1)

        x_win = transformed_data.reshape(-1, 3 * X.shape[1])

        return pd.DataFrame(x_win, columns=self.features)
