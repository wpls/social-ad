import gc
from time import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import load_npz
from sklearn.externals import joblib

import feature_construction as fc
import prepare as pp
import utilities as util
from predefine import *

# 缩写
# fe: Feature Engineering
# fc: feature construction
# f: feature
# fg: feature group


def one_hot():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart one hot')

    dataset = pd.read_hdf(path_intermediate_dataset + hdf_trainset_fg)

    # y
    y = dataset['label']
    del dataset['label']
    util.safe_save(path_modeling_dataset, npy_y, y)

    # 区分出类别特征
    categorical_features = ~dataset.columns.isin(numeric_features_set)

    # X
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categorical_features=categorical_features)
    X = enc.fit_transform(dataset.values)
    del dataset
    gc.collect()
    util.safe_save(path_modeling_dataset, npz_X, X)

    testset_ol = pd.read_hdf(path_intermediate_dataset + hdf_testset_ol_fg)

    # X_test_ol
    X_test_ol = enc.transform(testset_ol.values)
    del testset_ol
    gc.collect()
    util.safe_save(path_modeling_dataset, npz_X_test_ol, X_test_ol)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def split_train_test(train_proportion=0.8):
    """
    直接取后面 20% 的数据作为测试集
    
    Notes
    -----
    由于样本具有时序性，故不能使用 train_test_split 来随机划分，否则会导致数据泄露。
    """

    # 开始计时，并打印相关信息
    start = time()
    print('\nStart spliting train and test')

    # ===== X =====
    X = load_npz(path_modeling_dataset + npz_X)
    # 划分出训练集、测试集(注意不能随机划分)
    train_size = int(np.shape(X)[0] * train_proportion)
    # X_train
    X_train = X[:train_size, :]
    util.safe_save(path_modeling_dataset, npz_X_train, X_train)
    # X_test
    X_test = X[train_size:, :]
    util.safe_save(path_modeling_dataset, npz_X_test, X_test)
    # 手动释放内存
    del X

    # ===== y =====
    y = np.load(path_modeling_dataset + npy_y)
    # y_train
    y_train = y[:train_size]
    util.safe_save(path_modeling_dataset, npy_y_train, y_train)
    # y_test
    y_test = y[train_size:]
    util.safe_save(path_modeling_dataset, npy_y_test, y_test)
    # 手动释放内存
    del y
    gc.collect()

    # 停止计时，并打印相关信息
    util.print_stop(start)


def tuning_hyper_parameters():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters')

    # 加载训练集
    X_train = load_npz(path_modeling_dataset + npz_X_train)
    y_train = np.load(path_modeling_dataset + npy_y_train)

    from sklearn.metrics import make_scorer, log_loss
    loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # GridSearch
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import SGDClassifier
    alphas = np.logspace(-4, -1, 4)
    param_grid = {'alpha': alphas}
    generator = tscv.split(X_train)
    clf = GridSearchCV(SGDClassifier(loss='log', n_jobs=-1), param_grid, cv=generator, scoring=loss, n_jobs=-1)

    # 训练模型
    clf.fit(X_train, y_train)

    # 打印 cv_results
    cv_results_df = \
        DataFrame(clf.cv_results_)[['rank_test_score', 'param_alpha', 'mean_train_score', 'mean_test_score']]
    cv_results_df.rename(
        columns={'mean_train_score': 'mean_train_loss',
                 'mean_test_score': 'mean_val_loss',
                 'rank_test_score': 'rank_val_loss'},
        inplace=True)
    cv_results_df[['mean_val_loss', 'mean_train_loss']] = -cv_results_df[['mean_val_loss', 'mean_train_loss']]
    print('cv results: ')
    print(cv_results_df)

    # 手动释放内存
    del X_train
    del y_train
    gc.collect()

    # 加载测试集
    X_test = load_npz(path_modeling_dataset + npz_X_test)
    y_test = np.load(path_modeling_dataset + npy_y_test)
    # 打印在测试集上的 logloss
    print('logloss in testset: ', -clf.score(X=X_test, y=y_test))

    # 手动释放内存
    del X_test
    del y_test
    gc.collect()

    # 存储模型
    util.safe_save(path_model, 'sgd_lr.pkl', clf.best_estimator_)

    # 停止计时，并打印相关信息
    util.print_stop(start)
    
    
def tuning_hyper_parameters_sim():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters')

    # 加载训练集
    X_train = load_npz(path_modeling_dataset + npz_X)
    y_train = np.load(path_modeling_dataset + npy_y)    

    # 训练模型
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='log', alpha=0.01, n_jobs=-1)    
    clf.fit(X_train, y_train)
    
    # 打印在训练集上的 logloss
    from sklearn.metrics import log_loss
    print('logloss in trainset: ', log_loss(y_train, clf.predict_proba(X_train)))

    # 手动释放内存
    del X_train
    del y_train
    gc.collect()

    # 存储模型
    util.safe_save(path_model, 'sgd_lr.pkl', clf)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def predict_test_ol():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart predicting test_ol')

    # 加载 test_ol
    test_ol = pd.read_hdf(path_intermediate_dataset + hdf_test_ol)
    # # 加载 ad
    # ad = pd.read_hdf(path_intermediate_dataset + hdf_ad)
    # # 合并表格
    # test_ol = test_ol.merge(ad[['creativeID', 'appID']], how='left', on='creativeID')
    # # 构造 'userID-appID' 列
    # test_ol['userID-appID'] = test_ol['userID'].astype(str) + '-' + test_ol['appID'].astype(str)
    # # 加载已经有安装行为的 'userID-appID'
    # userID_appID_test = pd.read_hdf(path_intermediate_dataset + 'userID_appID_for_test.h5')

    # 加载 X_test_ol 和 model
    X_test_ol = load_npz(path_modeling_dataset + npz_X_test_ol)
    clf = joblib.load(path_model + 'sgd_lr.pkl')

    # 预测
    y_test_ol = clf.predict_proba(X_test_ol)

    # 生成提交数据集
    # submission = test_ol[['instanceID', 'label', 'userID-appID']].copy()
    submission = test_ol[['instanceID', 'label']].copy()
    submission.rename(columns={'label': 'prob'}, inplace=True)
    submission['prob'] = y_test_ol[:, 1]
    submission.set_index('instanceID', inplace=True)
    submission.sort_index(inplace=True)

    # # 对于那些已经有安装行为的 'userID-appID', 应该都预测为0
    # submission.loc[submission['userID-appID'].isin(userID_appID_test), 'prob'] = 0
    # # 删除 userID-appID 列
    # del submission['userID-appID']

    # 生成提交的压缩文件
    util.safe_save(path_submission_dataset, csv_submission, submission)

    # 停止计时，并打印相关信息
    util.print_stop(start)
