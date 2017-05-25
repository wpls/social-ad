import gc

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

    # 将地理位置调整到省级
    user_df['hometown'] = (user_df['hometown'] / 100).astype(int)
    user_df['residence'] = (user_df['residence'] / 100).astype(int)

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

    # 从 action 中提取出已经发生安装行为的 'userID_appID' 对
    action_df = pd.read_hdf(path_intermediate_dataset + hdf_action)
    indexer = action_df['userID'].isin(userID_set) & action_df['appID'].isin(appID_set)
    userID_appID_set |= set(util.elegant_pairing(action_df.loc[indexer, 'userID'],
                                                 action_df.loc[indexer, 'appID']))
    del action_df
    gc.collect()

    # 通过 list 转换为 Series 以存为 hdf5 格式
    util.safe_save(path_intermediate_dataset, hdf_userID_appID_pair_installed, Series(list(userID_appID_set)))

    # 停止计时，并打印相关信息
    util.print_stop(start)

    gc.collect()


def datatset():
    """
    为统一地构造 click_count, conversion_count, conversion_ratio 这三个特征准备 hdf_dataset.
    """

    out_file = path_intermediate_dataset + hdf_dataset
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_dataset)

    # ad
    app_cat_df = pd.read_hdf(path_intermediate_dataset + hdf_app_cat)
    ad_df = pd.read_hdf(path_intermediate_dataset + hdf_ad)
    ad_df = ad_df.merge(app_cat_df, how='left', on='appID')
    del app_cat_df
    gc.collect()

    # user
    user_df = pd.read_hdf(path_intermediate_dataset + hdf_user)

    # context_train
    pos_df = pd.read_hdf(path_intermediate_dataset + hdf_pos)
    context_train_df = pd.read_hdf(path_intermediate_dataset + hdf_train)
    context_train_df = context_train_df.merge(pos_df, how='left', on='positionID')
    del pos_df
    gc.collect()

    # dataset
    dataset_df = context_train_df.merge(user_df, how='left', on='userID')
    del context_train_df
    del user_df
    gc.collect()
    dataset_df = dataset_df.merge(ad_df, how='left', on='creativeID')
    del ad_df
    gc.collect()

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_dataset, dataset_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)

    gc.collect()


def prepare_dataset_all():
    """ 一次性执行所有的准备操作

    Notes
    -----

    """

    # 计时开始
    from time import time
    start = time()

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
    datatset()

    print('\nThe total time spent on preparing dataset: {0:.0f} s'.format(time() - start))
