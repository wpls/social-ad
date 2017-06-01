import gc
from time import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import load_npz
from sklearn.externals import joblib

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

    trainset_df = pd.read_hdf(path_feature + hdf_trainset_fg)
    if 'clickTime' in trainset_df.columns:
        del trainset_df['clickTime']

    # y
    y = trainset_df['label']
    del trainset_df['label']
    util.safe_save(path_modeling_dataset, npy_y, y)

    # 区分出类别特征
    numeric_features_s = pd.read_hdf(path_intermediate_dataset + hdf_numeric_features_set)
    numeric_features_set = set(numeric_features_s)
    categorical_features = \
        ~trainset_df.columns.isin(numeric_features_set | numeric_features_static_set | boolean_features_set)
    print('numeric_features: ', numeric_features_set | numeric_features_static_set)

    # X
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categorical_features=categorical_features)
    X = enc.fit_transform(trainset_df.values)
    del trainset_df
    gc.collect()
    util.safe_save(path_modeling_dataset, npz_X, X)

    testset_ol = pd.read_hdf(path_feature + hdf_testset_ol_fg)

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


def tuning_hyper_parameters_lr():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters of lr')

    # 加载训练集
    # X_train = load_npz(path_modeling_dataset + npz_X_train)
    # y_train = np.load(path_modeling_dataset + npy_y_train)
    X_train = load_npz(path_modeling_dataset + npz_X)
    y_train = np.load(path_modeling_dataset + npy_y)

    from sklearn.metrics import make_scorer, log_loss
    loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=10)

    # GridSearch
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import SGDClassifier
    # alphas = np.logspace(-6, -2, 5)
    alphas = [0.00006, 0.00008, 0.0001, 0.00012, 0.00014]
    param_grid = {'alpha': alphas}
    generator = tscv.split(X_train)
    clf = GridSearchCV(
        SGDClassifier(loss='log', random_state=42, n_jobs=-1),
        param_grid,
        cv=generator,
        scoring=loss,
        n_jobs=-1
    )

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

    # # 加载测试集
    # X_test = load_npz(path_modeling_dataset + npz_X_test)
    # y_test = np.load(path_modeling_dataset + npy_y_test)
    # # 打印在测试集上的 logloss
    # print('logloss in testset: ', -clf.score(X=X_test, y=y_test))

    # # 手动释放内存
    # del X_test
    # del y_test
    # gc.collect()

    # 存储模型, 方式一
    util.safe_save(path_model, 'sgd_lr.pkl', clf.best_estimator_)

    # # 以最佳参数在完整的数据集上重新训练， 方式二
    # best_clf = SGDClassifier(loss='log', alpha=clf.best_params_['alpha'], random_state=42, n_jobs=-1)
    # best_clf.fit(X_train, y_train)
    # util.safe_save(path_model, 'sgd_lr.pkl', best_clf)

    # 手动释放内存
    del X_train
    del y_train
    gc.collect()

    # 停止计时，并打印相关信息
    util.print_stop(start)


def tuning_hyper_parameters_xgb(train_proportion=0.1):
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters of xgb...')

    # 加载训练集
    trainset_df = pd.read_hdf(path_feature + 'fg_trainset.h5')

    # 划分训练集和线下测试集
    train_size = int(trainset_df.index.size * train_proportion)
    test_size = int(trainset_df.index.size * (train_proportion + 0.1))
    boolean_indexer_column = trainset_df.columns == 'label'

    y_train = trainset_df.loc[:train_size, boolean_indexer_column].values.ravel()
    y_test = trainset_df.loc[train_size:test_size, boolean_indexer_column].values.ravel()

    X_train = trainset_df.loc[:train_size, ~boolean_indexer_column].values
    X_test = trainset_df.loc[train_size:test_size, ~boolean_indexer_column].values

    del trainset_df
    gc.collect()

    # 时间序列划分
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)

    # 损失函数
    from sklearn.metrics import make_scorer, log_loss
    loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    # GridSearch
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    max_depth = [3, 4, 5]
    reg_alpha = [0, 0.0001, 0.001, 0.01, 0.1]
    reg_lambda = [0.1, 1, 10]
    subsample = [0.8, 0.9, 1]
    colsample_bytree = [0.8, 0.9, 1]
    colsample_bylevel = [0.8, 0.9, 1]
    param_grid = {
        'max_depth': max_depth,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel
    }
    generator = tscv.split(X_train)

    clf = GridSearchCV(
        XGBClassifier(n_estimators=125),
        param_grid,
        cv=generator,
        scoring=loss,
        n_jobs=-1
    )
    # 训练模型
    clf.fit(X_train, y_train)

    # 打印 cv_results
    cv_results_df = DataFrame(clf.cv_results_)
    cv_results_df.to_csv(path_cv_res + csv_cv_res_xgb)

    # 打印在训练集, 测试集上的 logloss
    from sklearn.metrics import log_loss
    print('logloss in trainset: ', log_loss(y_train, clf.predict_proba(X_train)))
    print('logloss in testset: ', log_loss(y_test, clf.predict_proba(X_test)))

    # 手动释放内存
    del X_train
    del y_train
    del X_test
    del y_test
    gc.collect()

    # 存储模型
    util.safe_save(path_model, 'xgb.pkl', clf.best_estimator_)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def tuning_hyper_parameters_lr_sim(n_iter_max = 100):
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters of lr_sim')

    # 加载训练集
    X = load_npz(path_modeling_dataset + npz_X).tocsr()
    # 划分出训练集、测试集(注意不能随机划分)
    train_size = 1208192
    # train_size = int(np.shape(X)[0] * train_proportion)
    # test_size = int(np.shape(X)[0] * (train_proportion + 0.1))
    # X_train
    X_train = X[:train_size, :]
    # X_test
    # X_test = X[train_size:test_size, :]
    X_test = X[train_size:, :]
    # 手动释放内存
    del X

    y = np.load(path_modeling_dataset + npy_y)
    # y_train
    y_train = y[:train_size]
    # y_test
    # y_test = y[train_size:test_size]
    y_test = y[train_size:]
    # 手动释放内存
    del y
    gc.collect()

    # 训练模型
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss
    loss_test_list = []
    clf_list = []
    alphas = np.logspace(-6, -2, 5)
    for alpha in alphas:
        clf = SGDClassifier(loss='log', alpha=alpha, n_jobs=-1)
        clf.fit(X_train, y_train)

        loss_train = log_loss(y_train, clf.predict_proba(X_train))
        loss_test = log_loss(y_test, clf.predict_proba(X_test))

        loss_test_list.append(loss_test)
        clf_list.append(clf)
        # 打印在训练集,测试集上的 logloss
        print('\nalpha = {0}'.format(alpha))
        print('logloss in trainset: ', loss_train)
        print('logloss in testset: ', loss_test)

    alpha = alphas[np.argmin(loss_test_list)]
    loss_test_prev = 1
    n_iter = 1
    clf = SGDClassifier(loss='log', alpha=alpha, n_iter=n_iter, n_jobs=-1, random_state=42)
    while n_iter < n_iter_max:
        clf = SGDClassifier(loss='log', alpha=alpha, n_iter=n_iter, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)

        loss_train = log_loss(y_train, clf.predict_proba(X_train))
        loss_test = log_loss(y_test, clf.predict_proba(X_test))

        print('\nn_iter = {0}'.format(n_iter))
        print('logloss in trainset: ', loss_train)
        print('logloss in testset: ', loss_test)

        if loss_test > loss_test_prev:
            break
        loss_test_prev = loss_test
        n_iter += 1

    # 手动释放内存
    del X_train
    del y_train
    del X_test
    del y_test
    gc.collect()

    # 存储模型
    util.safe_save(path_model, 'sgd_lr.pkl', clf)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def tuning_hyper_parameters_lr_sim_avg(train_proportion=0.8):
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters of lr sim avg')

    # 加载训练集
    X = load_npz(path_modeling_dataset + npz_X).tocsr()
    # 划分出训练集、测试集(注意不能随机划分)
    train_size = int(np.shape(X)[0] * train_proportion)
    test_size = int(np.shape(X)[0] * (train_proportion + 0.1))
    # X_train
    X_train = X[:train_size, :]
    # X_test
    X_test = X[train_size:test_size, :]
    # 手动释放内存
    del X

    y = np.load(path_modeling_dataset + npy_y)
    # y_train
    y_train = y[:train_size]
    # y_test
    y_test = y[train_size:test_size]
    # 手动释放内存
    del y
    gc.collect()

    # 训练模型
    from sklearn.linear_model import SGDClassifier
    clf1 = SGDClassifier(loss='log', alpha=0.0001, n_jobs=-1)
    clf2 = SGDClassifier(loss='log', alpha=0.0001, n_jobs=-1)
    clf3 = SGDClassifier(loss='log', alpha=0.0001, n_jobs=-1)

    from sklearn.ensemble import VotingClassifier
    eclf = VotingClassifier(estimators=[('lr1', clf1), ('lr2', clf2), ('lr3', clf3)], voting='soft')
    eclf.fit(X_train, y_train)
    
    # 打印在训练集，测试集上的 logloss
    from sklearn.metrics import log_loss
    print('logloss in trainset: ', log_loss(y_train, eclf.predict_proba(X_train)))
    print('logloss in testset: ', log_loss(y_test, eclf.predict_proba(X_test)))

    # 手动释放内存
    del X_train
    del y_train
    del X_test
    del y_test
    gc.collect()

    # 存储模型
    util.safe_save(path_model, 'sgd_lr.pkl', eclf)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def tuning_hyper_parameters_xgb_sim():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart tuning hyper parameters of xgb_sim...')

    # 加载训练集
    trainset_df = pd.read_hdf(path_feature + 'fg_trainset.h5')

    # 划分训练集和线下测试集
    row_indexer_train = (trainset_df['clickTime'] >= 210000) & (trainset_df['clickTime'] < 250000)
    row_indexer_test = (trainset_df['clickTime'] >= 250000) & (trainset_df['clickTime'] < 260000)

    del trainset_df['clickTime']
    col_indexer = trainset_df.columns == 'label'

    y_train = trainset_df.loc[row_indexer_train, col_indexer].values.ravel()
    y_test = trainset_df.loc[row_indexer_test, col_indexer].values.ravel()

    X_train = trainset_df.loc[row_indexer_train, ~col_indexer].values
    X_test = trainset_df.loc[row_indexer_test, ~col_indexer].values

    del trainset_df
    gc.collect()

    # 调参
    from xgboost import XGBClassifier
    from sklearn.metrics import log_loss
    loss_test_list = []
    clf_list = []
    for n in range(100, 220, 20):
        clf = XGBClassifier(n_estimators=n)
        clf.fit(X_train, y_train)

        loss_train = log_loss(y_train, clf.predict_proba(X_train))
        loss_test = log_loss(y_test, clf.predict_proba(X_test))

        loss_test_list.append(loss_test)
        clf_list.append(clf)
        # 打印在训练集,测试集上的 logloss
        print('\nn_estimators = {0}'.format(n))
        print('logloss in trainset: ', loss_train)
        print('logloss in testset: ', loss_test)

    # 手动释放内存
    del X_train
    del y_train
    del X_test
    del y_test
    gc.collect()

    # 存储模型
    util.safe_save(path_model, 'xgb.pkl', clf_list[np.argmin(loss_test_list)])

    # 停止计时，并打印相关信息
    util.print_stop(start)


def predict_test_ol_lr():
    # 开始计时，并打印相关信息
    start = time()
    print('\nStart predicting test_ol lr')

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
    util.safe_save(path_submission_dataset, csv_submission_lr, submission)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def predict_test_ol_xgb():
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
    X_test_ol = pd.read_hdf(path_feature + hdf_testset_ol_fg).values
    clf = joblib.load(path_model + 'xgb.pkl')

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
    util.safe_save(path_submission_dataset, csv_submission_xgb, submission)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def predict_average():
    """
    对 lr 的预测和 xgb 的预测做平均。
    :return:
    """

    # 开始计时，并打印相关信息
    start = time()
    print('\n对 lr 的预测和 xgb 的预测做平均。')

    submission_lr_series = pd.read_csv(path_submission_dataset + csv_submission_lr)
    submission_xgb_series = pd.read_csv(path_submission_dataset + csv_submission_xgb)

    submission_avg_series = (submission_lr_series + submission_xgb_series) / 2
    submission_avg_series['instanceID'] = submission_avg_series['instanceID'].astype(int)
    submission_avg_series.set_index('instanceID', inplace=True)

    util.safe_save(path_submission_dataset, csv_submission_avg, submission_avg_series)

    # 停止计时，并打印相关信息
    util.print_stop(start)


# test_results = pd.read_csv('test_results.csv')
def modelfit(clf, trainset_df, testset_df, feature_col, useTrainCV=False, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(trainset_df[feature_col].values, label=trainset_df[target].values)
        xgtest = xgb.DMatrix(testset_df[feature_col].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        clf.set_params(n_estimators=cvresult.shape[0])

    X_train = trainset_df[feature_col].values
    y_train = trainset_df['label'].values

    X_test = testset_df[feature_col].values
    y_test = testset_df['label'].values

    # 建模
    clf.fit(X_train, y_train, eval_metric='logloss')

    # 对训练集预测
    y_pred_train = clf.predict(X_train)
    y_pred_prob_train = clf.predict_proba(y_train)[:, 1]

    # 对测试集的预测
    y_pred_test = clf.predict(X_test)
    y_pred_prob_test = clf.predict_proba(y_test)[:, 1]

    # 输出模型的一些结果
    print("\n训练集")
    print("准确率 : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
    print("logloss 得分 (训练集): %f" % log_loss(y_train, y_pred_prob_train))

    print("\n测试集")
    print("准确率 : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
    print("logloss 得分 (训练集): %f" % log_loss(y_test, y_pred_prob_test))

    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance')
    plt.ylabel('Feature Importance Score')



