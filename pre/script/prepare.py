import gc
import os

import numpy as np
import pandas as pd
# from pandas import DataFrame
from pandas import Series

import utilities as util

from predefine import *


# from scipy.sparse import load_npz, save_npz, hstack, csr_matrix, csc_matrix

# 缩写
# fe: Feature Engineering
# fc: feature construction
# f: feature
# fg: feature group

# 数据准备，执行一次之后，很少再重复执行


def original_dataset():
    """
    如果 path_original_dataset 不存在，则创建该目录，同时下载'pre.zip'到该目录，并解压。
    """

    if not os.path.exists(path_original_dataset):
        import urllib.request
        urllib.request.urlretrieve(url_original_dataset, zip_original_dataset)

        from zipfile import ZipFile
        with ZipFile(zip_original_dataset, "r") as zip_ref:
            zip_ref.extractall(path=path_pre)

        # 重命名目录名称
        os.rename(path_pre + 'pre', path_original_dataset)


def transform_csv_to_hdf(csv, hdf):
    """

    :param csv: 
    :param hdf: 
    :return: 
    """
    out_file = path_intermediate_dataset + hdf
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf)

    in_file = path_original_dataset + csv
    df = pd.read_csv(in_file)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf, df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def ad():
    transform_csv_to_hdf(csv_ad, hdf_ad)


def app_cat():
    transform_csv_to_hdf(csv_app_cat, hdf_app_cat)


def pos():
    transform_csv_to_hdf(csv_pos, hdf_pos)


def test_ol():
    transform_csv_to_hdf(csv_test, hdf_test_ol)


def train():
    out_file = path_intermediate_dataset + hdf_train
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_train)

    # 加载 train.csv
    train_df = pd.read_csv(path_original_dataset + csv_train)

    # 原始的列，用于后面筛选
    original_columns = train_df.columns

    # ===== 填充telecomsOperator中的缺失值 =====
    userID_telecomsOperator = train_df.groupby(['userID', 'telecomsOperator'], as_index=False).count()
    userID_count = userID_telecomsOperator['userID'].value_counts()
    del userID_telecomsOperator
    gc.collect()

    userID_count_set_2 = set(userID_count.loc[userID_count == 2].index.values)
    del userID_count
    gc.collect()
    userID_missing_value_set = set(train_df.loc[train_df['telecomsOperator'] == 0, 'userID'])

    # 将缺失值置为NaN
    train_df.loc[train_df['telecomsOperator'] == 0, 'telecomsOperator'] = np.nan
    # 排序
    train_df.sort_values(by=['userID', 'telecomsOperator'], inplace=True)
    indexer = train_df['userID'].isin(userID_count_set_2 & userID_missing_value_set)
    del userID_count_set_2
    del userID_missing_value_set
    gc.collect()
    # 填充缺失值
    train_df.loc[indexer, 'telecomsOperator'] = train_df.loc[indexer, 'telecomsOperator'].ffill()

    # 将剩余的缺失值置为 0
    train_df['telecomsOperator'].fillna(value=0, inplace=True)

    # 重新以 clickTime 排序
    train_df.sort_values(by='clickTime', inplace=True)

    # ===== 删除不准确的样本 =====
    util.to_minute(train_df, 'clickTime')
    util.to_minute(train_df, 'conversionTime')

    train_df['deltaTime_min'] = train_df['conversionTime_min'] - train_df['clickTime_min']
    train_df['delta_deadline_min'] = 30 * 24 * 60 + 23 * 60 + 59 - train_df['clickTime_min']

    # 加载 ad.csv
    ad_df = pd.read_hdf(path_intermediate_dataset + 'ad.h5')
    train_df = train_df.merge(ad_df, how='left', on='creativeID')

    q = 0.91
    grouped = train_df[['advertiserID', 'deltaTime_min']].groupby('advertiserID', as_index=False)
    column = 'deltaTime_min_' + str(q)
    advertiserID_deltaTime_stats = grouped.quantile(q)
    advertiserID_deltaTime_stats.rename(columns={'deltaTime_min': column}, inplace=True)
    # 合并
    train_df = train_df.merge(advertiserID_deltaTime_stats, how='left', on='advertiserID')

    # 筛选, 仅保留有效行和原始列
    indexer_valid = (train_df['delta_deadline_min'] >= train_df[column]) | (train_df['label'] == 1)
    train_df = train_df.loc[indexer_valid, original_columns]

    # # 舍弃后一个小时的样本
    # train_df = train_df.loc[(train_df['clickTime'] <= 301220) & ((train_df['clickTime'] / 10000).astype(int) != 19)]

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_train, train_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)

    gc.collect()


def action():
    transform_csv_to_hdf(csv_action, hdf_action)


def user():
    out_file = path_intermediate_dataset + hdf_user
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_user)

    in_file = path_original_dataset + csv_user
    user_df = pd.read_csv(in_file)

    # # 将地理位置调整到省级
    # user_df['hometown'] = (user_df['hometown'] / 100).astype(int)
    # user_df['residence'] = (user_df['residence'] / 100).astype(int)

    # # 对 age 分段
    # age_interval = [0, 1, 4, 14, 29, 44, 59, 74, 84]
    # user_df['age'] = pd.cut(user_df['age'], age_interval, right=False, include_lowest=True, labels=False)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_user, user_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def user_app():
    transform_csv_to_hdf(csv_user_app, hdf_user_app)


def user_app_cat():
    out_file = path_intermediate_dataset + hdf_user_app_cat
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_user_app_cat)

    # 加载数据
    user_app_df = pd.read_hdf(path_intermediate_dataset + hdf_user_app)
    app_cat_df = pd.read_hdf(path_intermediate_dataset + hdf_app_cat)

    # 合并表格
    user_app_cat_df = user_app_df.merge(app_cat_df, on='appID', how='left')

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_user_app_cat, user_app_cat_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def userID_appID_pair_installed():
    """ 为训练集准备已存在安装行为的 'userID-appID'

    Notes
    -----
    该函数所生成的数据是给处理训练集时使用的，对于已存在安装行为的 'userID-appID'，其所
    对应的训练集中的样本应当直接舍弃。
    这样单独计算也是为了节省内存。因为train和test中的appID只占hdf_user_app中很少一部分
    """
    out_file = path_intermediate_dataset + hdf_userID_appID_pair_installed
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_userID_appID_pair_installed)

    # ===== train =====
    train_df = pd.read_hdf(path_intermediate_dataset + hdf_train)
    ad_df = pd.read_hdf(path_intermediate_dataset + hdf_ad)
    # 合并
    train_df = train_df.merge(ad_df, on='creativeID')
    # 单独提取出 userID, appID
    userID_set_train = set(train_df['userID'])
    appID_set_train = set(train_df['appID'])
    # 手动释放内存
    del train_df
    gc.collect()

    # ===== test_ol =====
    test_df = pd.read_hdf(path_intermediate_dataset + hdf_test_ol)
    # 合并
    test_df = test_df.merge(ad_df, on='creativeID')
    # 单独提取出 userID, appID
    userID_set_test_ol = set(test_df['userID'])
    appID_set_test_ol = set(test_df['appID'])
    # 手动释放内存
    del test_df
    del ad_df
    gc.collect()

    userID_set = userID_set_train | userID_set_test_ol
    appID_set = appID_set_train | appID_set_test_ol

    # 手动释放内存
    del userID_set_train
    del userID_set_test_ol
    del appID_set_train
    del appID_set_test_ol
    gc.collect()

    # 从 user_app 中提取出已经发生安装行为的 'userID_appID' 对
    user_app_df = pd.read_hdf(path_intermediate_dataset + hdf_user_app)
    indexer = user_app_df['userID'].isin(userID_set) & user_app_df['appID'].isin(appID_set)
    userID_appID_set = set(util.elegant_pairing(user_app_df.loc[indexer, 'userID'],
                                                user_app_df.loc[indexer, 'appID']))
    del user_app_df
    gc.collect()

    # 注：不能从action中直接提取，因为还与时间有关系
    # # 从 action 中提取出已经发生安装行为的 'userID_appID' 对
    # action_df = pd.read_hdf(path_intermediate_dataset + hdf_action)
    # indexer = action_df['userID'].isin(userID_set) & action_df['appID'].isin(appID_set)
    # userID_appID_set |= set(util.elegant_pairing(action_df.loc[indexer, 'userID'],
    #                                              action_df.loc[indexer, 'appID']))
    # del action_df
    # gc.collect()

    # 通过 list 转换为 Series 以存为 hdf5 格式
    util.safe_save(path_intermediate_dataset, hdf_userID_appID_pair_installed, Series(list(userID_appID_set)))

    # 停止计时，并打印相关信息
    util.print_stop(start)

    gc.collect()


def datatset(hdf_out, hdf_in):
    """
    为统一地构造 click_count, conversion_count, conversion_ratio 这三个特征准备数据集.
    """

    out_file = path_intermediate_dataset + hdf_out
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_out)

    # ad
    app_cat_df = pd.read_hdf(path_intermediate_dataset + hdf_app_cat)
    ad_df = pd.read_hdf(path_intermediate_dataset + hdf_ad)
    ad_df = ad_df.merge(app_cat_df, how='left', on='appID')
    del app_cat_df
    gc.collect()

    # user
    user_df = pd.read_hdf(path_intermediate_dataset + hdf_user)

    # context
    pos_df = pd.read_hdf(path_intermediate_dataset + hdf_pos)
    context_df = pd.read_hdf(path_intermediate_dataset + hdf_in)
    context_df = context_df.merge(pos_df, how='left', on='positionID')
    del pos_df
    gc.collect()

    # dataset
    dataset_df = context_df.merge(user_df, how='left', on='userID')
    del context_df
    del user_df
    gc.collect()
    dataset_df = dataset_df.merge(ad_df, how='left', on='creativeID')
    del ad_df
    gc.collect()

    # 准备 hour, week
    dataset_df['hour'] = (dataset_df['clickTime'] / 100 % 100).astype(int)
    dataset_df['week'] = (dataset_df['clickTime'] / 10000).astype(int) % 7

    # 如果条件满足，则舍弃后 5 天的负样本
    if 'train' in hdf_in and discard_negative_last_5_day:
        dataset_df = dataset_df.loc[(dataset_df['clickTime'] < 260000) | (dataset_df['label'] != 0)]

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_out, dataset_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)

    gc.collect()


def trainset():
    datatset(hdf_trainset, hdf_train)


def testset_ol():
    datatset(hdf_testset_ol, hdf_test_ol)


def prepare_dataset_all():
    """ 一次性执行所有的准备操作

    Notes
    -----

    """

    # 计时开始
    from time import time
    start = time()

    original_dataset()

    ad()
    app_cat()
    pos()
    test_ol()
    train()
    action()
    user()
    user_app()
    user_app_cat()
    userID_appID_pair_installed()
    trainset()
    testset_ol()

    print('\nThe total time spent on preparing dataset: {0:.0f} s'.format(time() - start))
