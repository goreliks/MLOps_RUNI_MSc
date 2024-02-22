import sys
import os
import pandas as pd
import numpy as np
import math
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from algorithms.MSET import MSET
from tsad.evaluating.evaluating import evaluating
from transformers.shap_feature_selector import SHAPFeatureSelector
from config import F_SCORE_BETA, LOOK_BACK, DATA_PATH, FEATURE_IMPORTANCE, IMPROVEMENT_PIPELINE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

INITIAL_FEATURE_NAMES = []

predicted_outlier = []
rel_errors = []

predicted_outlier_improved = []
rel_errors_improved = []


def load_data(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('csv'):
                all_files.append(os.path.join(root, filename))

    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file,
                              sep=';',
                              index_col='datetime',
                              parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    # anomaly-free df loading
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0],
                                  sep=';',
                                  index_col='datetime',
                                  parse_dates=True)

    global INITIAL_FEATURE_NAMES
    INITIAL_FEATURE_NAMES = list_of_df[0].drop(['anomaly', 'changepoint'], axis=1).columns

    # dataset characteristics printing

    print('The SKAB v0.9 dataset loaded:')
    print(f'- A number of datasets: {len(list_of_df)}')
    print(f'- Shape of the random dataset: {list_of_df[10].shape}')
    n_outlier = sum([len(df[df.anomaly == 1.]) for df in list_of_df])
    print(f'- A number of outliers (point anomalies): {n_outlier}\n')
    print(f'Head of the random dataset:')
    print(list_of_df[0].head(1))

    return list_of_df


def move_mean(array, window):
    n = np.size(array)
    xx = array.copy()
    y = []
    for i in range(0, window):
        y.append(np.roll(xx.tolist() + [np.nan]*window, i))
    y = np.nanmean(y, axis=0)
    l = math.ceil(window/2)

    return y[l-1:n+l-1]


def mset_prediction_wrapper(X_array, improved=False):
    """
    This function wraps around the MSET prediction logic to return binary anomaly predictions.
    X_array is expected to be a 2D numpy array where each row is a sample.
    """
    # Ensure the input is a DataFrame with the correct column names
    if not improved:
        X_df = pd.DataFrame(X_array, columns=INITIAL_FEATURE_NAMES)
    else:
        X_df = pd.DataFrame(X_array, columns=X_array.columns)
    ms = MSET()

    ms.fit(X_df[:400])
    # Predict using MSET model
    Y_pred = ms.predict(X_df)

    # Calculate errors and relative errors
    err = np.linalg.norm(X_df.values - Y_pred.values, axis=1)
    rel_err = move_mean(err / np.linalg.norm(Y_pred.values, axis=1), window=60)
    if improved:
        rel_errors_improved.append(rel_err)
    else:
        rel_errors.append(rel_err)

    # Determine binary predictions based on the relative error threshold
    predictions = (rel_err > 0.01).astype(int)
    if improved:
        predicted_outlier_improved.append(pd.DataFrame((rel_err > 0.01), X_df.index).fillna(0).any(axis=1).astype(int))
    else:
        predicted_outlier.append(pd.DataFrame((rel_err > 0.01), X_df.index).fillna(0).any(axis=1).astype(int))

    return predictions


def print_out_results(true_outliers, predicted_outliers, improved=False):
    test_cm = confusion_matrix(true_outliers, predicted_outliers)
    tn, fp, fn, tp = test_cm.ravel()

    test_acc = accuracy_score(true_outliers, predicted_outliers)
    test_f_beta_score = fbeta_score(true_outliers.values, predicted_outliers.values, beta=F_SCORE_BETA)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    pipe_type = 'BASELINE'
    if improved:
        pipe_type = 'IMPROVED'

    print("******************************")
    print(f"{pipe_type} PIPELINE PERFORMANCE:")
    print("******************************")
    print(f'ACCURACY: {test_acc:.3f}')
    print(f'PRECISION: {precision:.3f}')
    print(f'RECALL: {recall:.3f}')
    print(f'F_BETA SCORE: {test_f_beta_score:.3f}')
    print('CONFUSION MATRIX:')
    print(test_cm)


if __name__ == '__main__':
    list_of_df = load_data(DATA_PATH)
    print('=========================================')
    true_outlier = [df.anomaly for df in list_of_df]
    full_true_outlier = pd.concat(true_outlier)
    if not IMPROVEMENT_PIPELINE:
        print('The training of a baseline pipeline is in process...')
        for data_frame in list_of_df:
            mset_prediction_wrapper(data_frame.drop(['anomaly', 'changepoint'], axis=1))
        full_predicted_outlier = pd.concat(predicted_outlier)
        print('=========================================')
        print_out_results(full_true_outlier, full_predicted_outlier)
    else:
        shap_feature_selector_f_beta = SHAPFeatureSelector(importance=FEATURE_IMPORTANCE, estimator=mset_prediction_wrapper, beta=F_SCORE_BETA)
        print('The training of the improved pipeline is in process...')
        for i, date_frame in enumerate(list_of_df):
            print(f'Starting the training on data frame #{i+1} out of {len(list_of_df)}:')
            train_x, train_y = date_frame.drop(['anomaly', 'changepoint'], axis=1), date_frame['anomaly']
            train_x_imp_beta = shap_feature_selector_f_beta.fit_transform(train_x, train_y.values)
            mset_prediction_wrapper(train_x_imp_beta, improved=True)
        full_predicted_outlier_improved = pd.concat(predicted_outlier_improved)
        print('=========================================')
        print_out_results(full_true_outlier, full_predicted_outlier_improved, improved=True)
