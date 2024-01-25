import os
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, fbeta_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import random
import tensorflow as tf
from sklearn.pipeline import Pipeline
from transformers.kaiser_window_smoother import KaiserWindowSmoother
from transformers.temporal_feature_creator import TemporalFeatureCreator
from transformers.shap_feature_selector import SHAPFeatureSelector
from sklearn.preprocessing import StandardScaler
from config import F_SCORE_BETA, LOOK_BACK, DATA_PATH, FEATURE_IMPORTANCE, IMPROVEMENT_PIPELINE
import warnings
warnings.filterwarnings('ignore')


def load_and_split_data(directory, train_proportion=0.7,
                        valid_proportion=0.66):  # valid_size adjusted for 70% train, 20% valid, 10% test
    all_files = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('csv'):
                all_files.append(f'{dirname}/{filename}')
    all_files.sort()

    valve1_dat = {file.split('/')[-1]: pd.read_csv(file, sep=';', index_col='datetime', parse_dates=True)
                  for file in all_files if 'valve1' in file}

    # concatenate data(order in time series by sort_index)
    valve1_data = pd.concat(list(valve1_dat.values()), axis=0).sort_index()

    train_pre_size = len(valve1_data)
    train_size = int(train_pre_size * train_proportion)
    train = valve1_data[0:train_size]
    y_train = valve1_data[0:train_size]['anomaly'].values
    x_train = valve1_data[0:train_size].drop(columns=['anomaly', 'changepoint'])

    valid_pre_size = train_pre_size - train_size
    valid_size = int(valid_pre_size * valid_proportion)
    valid = valve1_data[train_size:train_size + valid_size]
    y_valid = valve1_data[train_size:train_size + valid_size]['anomaly'].values
    x_valid = valve1_data[train_size:train_size + valid_size].drop(columns=['anomaly', 'changepoint'])

    test = valve1_data[train_size + valid_size:]
    y_test = valve1_data[train_size + valid_size:]['anomaly'].values
    x_test = valve1_data[train_size + valid_size:].drop(columns=['anomaly', 'changepoint'])

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def lgb_train_predict(x_train, y_train, x_valid, y_valid, x_test, y_test, beta=0.5):

    # fix random seed
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"

    # fine-tunned hyper parameters as found in baseline
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'force_row_wise': True,
        'seed': 0,
        'learning_rate': 0.0424127,
        'min_data_in_leaf': 15,
        'max_depth': 24,
        'num_leaves': 29
    }

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid)
    lgb_test = lgb.Dataset(x_test, y_test)

    model_lgb = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid],
                          verbose_eval=0, early_stopping_rounds=20)

    test_pred = np.zeros((len(y_test), 1))
    test_pred[:, 0] = np.where(model_lgb.predict(x_test) >= 0.5, 1, 0)
    test_acc = accuracy_score(y_test.reshape(-1, 1), test_pred)
    test_f1score = f1_score(y_test.reshape(-1, 1), test_pred)
    test_cm = confusion_matrix(y_test.reshape(-1, 1), test_pred)
    test_f_beta_score = fbeta_score(y_test.reshape(-1, 1), test_pred, beta=beta)

    return test_acc, test_f1score, test_cm, test_pred, model_lgb, test_f_beta_score


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_and_split_data(DATA_PATH)

    pipeline = Pipeline([
        ('look_back', KaiserWindowSmoother()),
        ('scaler', StandardScaler()),
        ('classifier', TemporalFeatureCreator(LOOK_BACK))
    ])

    train_x = pipeline.fit_transform(x_train)
    valid_x = pipeline.transform(x_valid)
    test_x = pipeline.transform(x_test)

    train_y = y_train[LOOK_BACK - 1:]
    valid_y = y_valid[LOOK_BACK - 1:]
    test_y = y_test[LOOK_BACK - 1:]

    test_acc, test_f1score, test_cm, test_pred, model_lgb, test_f_beta_score = lgb_train_predict(train_x, train_y,
                                                                                                 valid_x, valid_y,
                                                                                                 test_x, test_y,
                                                                                                 beta=F_SCORE_BETA)
    if not IMPROVEMENT_PIPELINE:
        print("******************************")
        print("BASELINE PIPELINE PERFORMANCE:")
        print("******************************")
        print(f'TEST ACCURACY: {test_acc:.3f}')
        print(f'TEST F1 SCORE: {test_f1score:.3f}')
        print(f'TEST F_BETA SCORE: {test_f_beta_score:.3f}')
        print('TEST CONFUSION MATRIX:')
        print(test_cm)
    else:
        shap_feature_selector_f_beta = SHAPFeatureSelector(importance=FEATURE_IMPORTANCE,
                                                           estimator=model_lgb, beta=F_SCORE_BETA)
        train_x_imp_beta = shap_feature_selector_f_beta.fit_transform(train_x, train_y)
        valid_x_imp_beta = shap_feature_selector_f_beta.transform(valid_x)
        test_x_imp_beta = shap_feature_selector_f_beta.transform(test_x)

        test_acc_imp, test_f1score_imp, test_cm_imp, test_pred_imp, model_lgb_imp, test_f_beta_score_imp = (
            lgb_train_predict(train_x_imp_beta, train_y, valid_x_imp_beta,
                              valid_y, test_x_imp_beta, test_y, beta=F_SCORE_BETA))

        print("******************************")
        print("IMPROVED PIPELINE PERFORMANCE:")
        print("******************************")
        print(f'TEST ACCURACY: {test_acc_imp:.3f}')
        print(f'TEST F1 SCORE: {test_f1score_imp:.3f}')
        print(f'TEST F_BETA SCORE: {test_f_beta_score_imp:.3f}')
        print('TEST CONFUSION MATRIX:')
        print(test_cm_imp)




