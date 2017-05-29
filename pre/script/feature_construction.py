import gc
import os
from time import time

import numpy as np
import pandas as pd
from pandas import DataFrame

import utilities as util
from predefine import *

"""  
缩写
fe: feature engineering
fc: feature construction
 f: feature
fg: feature group
"""


# ========== feature construction ad ==========

def f_app_popularity():
    out_file = path_feature + hdf_app_popularity
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_app_popularity)

    # 加载 user_app
    user_app = pd.read_hdf(path_intermediate_dataset + hdf_user_app)

    # 提取 app 的热度特征
    app_popularity = user_app.groupby('appID', as_index=False).count()
    app_popularity.rename(columns={'userID': fn_app_popularity}, inplace=True)

    # 手动释放内存
    del user_app
    gc.collect()

    # 存储
    util.safe_save(path_feature, hdf_app_popularity, app_popularity)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def fg_ad():
    out_file = path_intermediate_dataset + hdf_ad_fg
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_ad_fg)

    # 加载 ad.h5 和 app_cat.h5
    ad = pd.read_hdf(path_intermediate_dataset + hdf_ad)
    app_cat = pd.read_hdf(path_intermediate_dataset + hdf_app_cat)

    # 合并 ad 和 app_cat
    ad = ad.merge(app_cat, on='appID')

    # 手动释放内存
    del app_cat
    gc.collect()

    # 加载 app_popularity.h5
    in_file = path_intermediate_dataset + hdf_app_popularity
    if not os.path.exists(in_file):
        f_app_popularity()
    app_popularity = pd.read_hdf(in_file)

    # 合并表格
    # 因为构造该特征时使用了 groupby，故此时的 index 是 'appID', 需要在合并之前将其恢复为 columns
    app_popularity.reset_index(inplace=True)
    ad = ad.merge(app_popularity, how='left', on='appID')

    # 手动释放内存
    del app_popularity
    gc.collect()

    # 将 app_popularity 离散化
    ad[fn_app_popularity] = pd.cut(ad[fn_app_popularity], np.logspace(0, 7, num=8), include_lowest=True, labels=False)

    # 将 app_popularity 的 NaN 填充为 6
    ad[fn_app_popularity].fillna(6, inplace=True)

    # 提取出部分特征
    selected_feature = ['creativeID',
                        # 'adID', \
                        # 'camgaignID', \
                        'advertiserID',
                        # 'appID',
                        'appPlatform',
                        'appCategory',
                        'app_popularity'
                        ]
    ad_selected = ad[selected_feature]

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_ad_fg, ad_selected)

    del ad
    gc.collect()

    # 停止计时，并打印相关信息
    util.print_stop(start)


# ========== feature construction user ==========

def f_user_activity():
    out_file = path_feature + hdf_user_activity
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_user_activity)

    # 提取用户的活跃度特征
    user_app = pd.read_hdf(path_intermediate_dataset + hdf_user_app)
    user_activity = user_app.groupby('userID').count()
    user_activity.rename(columns={'appID': fn_user_activity}, inplace=True)

    # 手动释放内存
    del user_app
    gc.collect()

    # 离散化
    interval = np.ceil(np.logspace(0, 3, 6))
    user_activity[fn_user_activity] = \
        pd.cut(user_activity[fn_user_activity], interval, include_lowest=True, labels=False)
    user_activity.reset_index(inplace=True)

    # 存储
    util.safe_save(path_feature, hdf_user_activity, user_activity)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def f_user_pref_cat():
    out_file = path_feature + hdf_user_pref_cat
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_user_pref_cat)

    # 加载数据
    user_app_cat = pd.read_hdf(path_intermediate_dataset + hdf_user_app_cat)

    # 计算同一用户安装相同品类app的数量
    user_cat_count = user_app_cat.groupby(['userID', 'appCategory'], as_index=False).count()
    user_cat_count.rename(columns={'appID': 'count'}, inplace=True)

    # 手动释放内存
    del user_app_cat
    gc.collect()

    # 提取数量最多的非未知品类，作为用户的偏好品类
    group_idxmax = \
        user_cat_count.loc[user_cat_count['appCategory'] != 0, ['userID', 'count']].groupby('userID').idxmax()
    user_pref_cat = user_cat_count.loc[group_idxmax['count'], ['userID', 'appCategory']]
    user_pref_cat.rename(columns={'appCategory': fn_cat_pref}, inplace=True)

    # 存储
    util.safe_save(path_feature, hdf_user_pref_cat, user_pref_cat)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def fg_user():
    out_file = path_intermediate_dataset + hdf_user_fg
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_user_fg)

    # 加载 user 数据
    user = pd.read_hdf(path_intermediate_dataset + hdf_user)

    # 加载并添加用户的活跃度特征
    in_file = path_intermediate_dataset + hdf_user_activity
    if not os.path.exists(in_file):
        f_user_activity()
    user_activity = pd.read_hdf(in_file)
    user = user.merge(user_activity, how='left', on='userID')

    # 手动释放内存
    del user_activity
    gc.collect()

    # 将 user_activity 的 NaN 填充为 5
    user[fn_user_activity].fillna(5, inplace=True)

    # 加载并添加用户对app的品类偏好特征
    in_file = path_intermediate_dataset + hdf_user_pref_cat
    if not os.path.exists(in_file):
        f_user_pref_cat()
    user_pref_cat = pd.read_hdf(in_file)
    user = user.merge(user_pref_cat, how='left', on='userID')

    # 手动释放内存
    del user_pref_cat
    gc.collect()

    # 将 cat_pref 的 NaN 填充为 0
    user[fn_cat_pref].fillna(0, inplace=True)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_user_fg, user)

    # 停止计时，并打印相关信息
    util.print_stop(start)


# ========== feature construction context ==========

# 暂时没用
def f_hour_weight():
    out_file = path_intermediate_dataset + hdf_hour_weight
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_hour_weight)

    # 加载 action.h5
    action = pd.read_hdf(path_intermediate_dataset + 'action.h5')

    # 提取每小时的行为数据
    hour_weight = (action['installTime'] / 100 % 100).astype(int).value_counts()
    hour_weight_df = DataFrame(hour_weight).reset_index()
    hour_weight_df.rename(columns={'index': 'hour', 'installTime': 'hour_weight'}, inplace=True)

    # 手动释放内存
    del action
    gc.collect()

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_hour_weight, hour_weight_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def f_userID():
    out_file = path_intermediate_dataset + hdf_userID
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_userID)

    # 加载 train.h5
    train_df = pd.read_hdf(path_intermediate_dataset + 'train.h5')

    # 从`userID`中提取`conversion_count`特征
    userID_count_positive = train_df.loc[train_df['label'] == 1, 'userID'].value_counts()
    userID_count_positive.sort_index(inplace=True)

    userID_count_positive_df = DataFrame(userID_count_positive)
    userID_count_positive_df.reset_index(inplace=True)
    userID_count_positive_df.columns = ['userID', 'conversion_count']

    del userID_count_positive
    gc.collect()

    # 对`userID`提取`click_count_group`特征
    userID_count = train_df['userID'].value_counts()
    userID_count.sort_index(inplace=True)

    userID_count_df = DataFrame(userID_count)
    userID_count_df.reset_index(inplace=True)
    userID_count_df.columns = ['userID', 'click_count']

    del userID_count
    gc.collect()

    # 对 click_count 分组
    bins = [1, 28, 44, 120]
    userID_count_df['click_count_group'] = \
        pd.cut(userID_count_df['click_count'], bins=bins, include_lowest=True, labels=False)

    # 合并
    f_userID_df = userID_count_df.merge(userID_count_positive_df, how='left', on='userID')
    del userID_count_df
    del userID_count_positive_df
    gc.collect()

    # 将缺失值填充为0
    f_userID_df['conversion_count'].fillna(value=0, inplace=True)

    # 对`userID`提取`conversion_ratio`特征
    f_userID_df['conversion_ratio_click'] = f_userID_df['conversion_count'] / f_userID_df['click_count']

    # 存储
    del f_userID_df['click_count']
    util.safe_save(path_intermediate_dataset, hdf_userID, f_userID_df)

    # 手动释放内存
    del train_df
    gc.collect()

    # 停止计时，并打印相关信息
    util.print_stop(start)

"""
def fg_context(hdf_out, hdf_in):
    out_file = path_intermediate_dataset + hdf_out
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_out)

    # 加载 hdf_in
    df = pd.read_hdf(path_intermediate_dataset + hdf_in)

    # 添加 connectionType 的转化率特征
    in_file = path_intermediate_dataset + hdf_conversion_ratio_connectionType
    if not os.path.exists(in_file):
        # f_conversion_ratio_connectionType()
    conversion_ratio_connectionType = pd.read_hdf(in_file)
    df = df.merge(conversion_ratio_connectionType, how='left', on='connectionType')
    print('添加 connectionType 的转化率特征', df.shape)
    del conversion_ratio_connectionType

    # 添加 telecomsOperator 的转化率特征
    in_file = path_intermediate_dataset + hdf_conversion_ratio_telecomsOperator
    if not os.path.exists(in_file):
        f_conversion_ratio_telecomsOperator()
    conversion_ratio_telecomsOperator = pd.read_hdf(in_file)
    df = df.merge(conversion_ratio_telecomsOperator, how='left', on='telecomsOperator')
    print('添加 telecomsOperator 的转化率特征', df.shape)
    del conversion_ratio_telecomsOperator

    # 提取 hour 特征
    df['hour'] = (df['clickTime'] / 100 % 100).astype(int)

    # 添加与`hour`相关的特征
    in_file = path_intermediate_dataset + hdf_hour
    if not os.path.exists(in_file):
        f_hour()
    f_hour_df = pd.read_hdf(in_file)
    df = df.merge(f_hour_df, how='left', on='hour')
    print('添加与`hour`相关的特征', df.shape)
    del f_hour_df
    gc.collect()

    # 提取 week 特征
    df['week'] = (df['clickTime'] / 10000).astype(int) % 7

    # 添加与`week`相关的特征
    in_file = path_intermediate_dataset + hdf_week
    if not os.path.exists(in_file):
        f_week()
    f_week_df = pd.read_hdf(in_file)
    df = df.merge(f_week_df, how='left', on='week')
    print('添加与`week`相关的特征', df.shape)
    del f_week_df
    gc.collect()

    # 添加与`userID`相关的特征
    in_file = path_intermediate_dataset + hdf_userID
    if not os.path.exists(in_file):
        f_userID()
    f_userID_df = pd.read_hdf(in_file)
    df = df.merge(f_userID_df, on='userID', how='left')
    print('添加与`userID`相关的特征', df.shape)
    del f_userID_df
    gc.collect()
    # 对线上测试集来说，此时会出现缺失值
    if 'test' in hdf_in:
        # 将缺失的conversion_count填充为0
        df['conversion_count'].fillna(value=0, inplace=True)
        # 将缺失的click_count_group填充为0
        df['click_count_group'].fillna(value=0, inplace=True)
        # 将缺失的 conversion_ratio_click 填充为0
        df['conversion_ratio_click'].fillna(value=0, inplace=True)

    # ===== 添加“该 userID_appID 是否已存在安装行为”的特征 =====
    # 加载 ad.h5
    ad_df = pd.read_hdf(path_intermediate_dataset + hdf_ad)

    # 合并 train, ad
    df = df.merge(ad_df[['creativeID', 'appID']], how='left', on='creativeID')
    print('合并 ad', df.shape)
    del ad_df
    gc.collect()

    # # 构造 'userID-appID' 列, 有没有更好的编码方式？str太占内存了。有，接下来看寡人的牛逼函数！
    # df['userID-appID'] = df['userID'].astype(str) + '-' + df['appID'].astype(str)

    df['userID-appID'] = util.elegant_pairing(df['userID'], df['appID'])
    del df['appID']
    gc.collect()

    # 加载 userID_appID.h5
    userID_appID = pd.read_hdf(path_intermediate_dataset + hdf_userID_appID_pair_installed)

    # 只保留没有安装行为的 userID_appID（这个操作也应当放到数据清洗里）,还是提取为一个特征比较好
    # df = df.loc[~df['userID-appID'].isin(userID_appID)]
    df[fn_is_installed] = df['userID-appID'].isin(userID_appID)

    # 释放内存
    del df['userID-appID']
    del userID_appID
    gc.collect()

    # 加载 pos.h5
    pos_df = pd.read_hdf(path_intermediate_dataset + hdf_pos)
    # 合并表格
    df = df.merge(pos_df, on='positionID', how='left')
    print('合并 pos', df.shape)
    del pos_df
    gc.collect()

    # 准备存储
    if 'train' in hdf_in:
        # 舍弃后5天的负样本(感觉这个操作应该放到数据清洗里)
        # 这里会用到clickTime，所以在此之前，不能删除
        df = df.loc[(df['clickTime'] < 260000) | (df['label'] != 0)]
        del df['clickTime']
        del df['conversionTime']
        del df['positionID']
    elif 'test' in hdf_in:
        del df['instanceID']
        del df['label']
        del df['clickTime']
        del df['positionID']

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_out, df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def fg_context_dataset():
    fg_context(hdf_context_dataset_fg, hdf_train)


def fg_context_testset_ol():
    fg_context(hdf_context_testset_ol_fg, hdf_test_ol)
"""


# ========== feature construction ==========
def f_confidence_trainset(trainset_df):
    """
    为trainset构造样本的置信度特征。
    """

    # 将时间转换为以分钟为单位
    util.to_minute(trainset_df, 'clickTime')
    util.to_minute(trainset_df, 'conversionTime')
    # 构造时间差
    trainset_df['deltaTime_min'] = trainset_df['conversionTime_min'] - trainset_df['clickTime_min']
    # 分析时间差的分布
    distribution_deltaTime = trainset_df['deltaTime_min'].value_counts()
    distribution_deltaTime.sort_index(inplace=True)
    distribution_deltaTime_df = DataFrame(distribution_deltaTime)
    distribution_deltaTime_df.reset_index(inplace=True)
    distribution_deltaTime_df.columns = ['deltaTime_min', 'num']
    del distribution_deltaTime
    gc.collect()
    # 提取回流率与时间差大小的关系
    distribution_deltaTime_df['ratio'] = distribution_deltaTime_df['num'] / distribution_deltaTime_df['num'].sum()
    distribution_deltaTime_df['ratio_cumsum'] = distribution_deltaTime_df['ratio'].cumsum()
    """
    如何从这个表中获取样本的置信度  
    1. testset_ol 中样本的置信度都为1
    2. 所有正样本的置信度都为1
    3. 训练集中负样本的置信度与距302359的分钟时间差有关系。
       - 距离截止时间7153分钟以上的样本置信度为1；
       - 7153分钟以内的，根据距302359的分钟时间差deltaTime_min在上表中查询。
    """

    # 提取 confidence_df
    distribution_deltaTime_df['deltaTime_min'] = distribution_deltaTime_df['deltaTime_min'].astype(int, copy=False)
    distribution_deltaTime_df.set_index('deltaTime_min', inplace=True)

    new_index = list(range(distribution_deltaTime_df.index.max() + 1))
    confidence = distribution_deltaTime_df['ratio_cumsum'].reindex(new_index, method='ffill')

    confidence.rename('confidence', inplace=True)
    confidence.index.name = 'delta_deadline_min'
    confidence_df = confidence.reset_index()

    # 获取最后五天样本的置信度
    trainset_df['delta_deadline_min'] = 30 * 24 * 60 + 23 * 60 + 59 - trainset_df['clickTime_min']
    trainset_df = trainset_df.merge(confidence_df, how='left', on='delta_deadline_min')

    # 所有正样本的置信度都为1, 距离截止时间7153分钟以上的样本置信度为1
    indexer = (trainset_df['label'] == 1) | (trainset_df['delta_deadline_min'] > 7153)
    trainset_df.loc[indexer, 'confidence'] = 1

    # 删除多余的列
    del trainset_df['clickTime_min']
    del trainset_df['conversionTime_min']
    del trainset_df['deltaTime_min']
    del trainset_df['delta_deadline_min']

    return trainset_df


def f_confidence_testset_ol(testset_ol):
    """
        为 testset_ol 构造样本的置信度特征。
    """
    testset_ol['confidence'] = 1
    return testset_ol


def f_is_installed_from_action_trainset(trainset_df):
    """
    根据 action 数据构造 fn_is_installed 特征。
    :param trainset_df:
    :return:
    """

    action_df = pd.read_hdf(path_intermediate_dataset + hdf_action)
    action_df['userID_appID'] = util.elegant_pairing(action_df['userID'], action_df['appID'])

    # 判断 trainset_df 中是否存在 userID_appID 列
    if 'userID_appID' not in trainset_df.columns:
        trainset_df['userID_appID'] = util.elegant_pairing(trainset_df['userID'], trainset_df['appID'])

    # 合并
    trainset_df = trainset_df.merge(action_df[['userID_appID', 'installTime']], how='left', on='userID_appID')

    trainset_df[fn_is_installed] = trainset_df['clickTime'] > trainset_df['installTime']

    del trainset_df['installTime']
    del action_df
    gc.collect()

    return trainset_df


def f_is_installed_from_action_testset_ol(testset_ol_df):
    """
    根据 action 数据构造 fn_is_installed 特征。
    :param testset_ol_df:
    :return:
    """

    action_df = pd.read_hdf(path_intermediate_dataset + hdf_action)
    action_df['userID_appID'] = util.elegant_pairing(action_df['userID'], action_df['appID'])

    # 判断 testset_ol_df 中是否存在 userID_appID 列
    if 'userID_appID' not in testset_ol_df.columns:
        testset_ol_df['userID_appID'] = util.elegant_pairing(testset_ol_df['userID'], testset_ol_df['appID'])

    testset_ol_df[fn_is_installed] = testset_ol_df['userID_appID'].isin(action_df['userID_appID'])

    del action_df
    gc.collect()

    return testset_ol_df


def f_count_ratio():
    """
    为 hdf_datatset 中的特征构造 click_count, conversion_count, conversion_ratio 特征，并存储到硬盘。
    :return: 
    """

    # 加载数据集
    trainset_df = pd.read_hdf(path_intermediate_dataset + hdf_trainset)

    # 安全的将 hdf_numeric_features_set 清空, 以避免可能的重复添加
    util.safe_remove(path_intermediate_dataset + hdf_numeric_features_set)
    # 遍历数据集中的有效特征
    for c in trainset_df.columns:
        if c not in (columns_set_without_count_ratio | columns_set_mismatch):
            util.f_count_ratio(trainset_df, c)

    del trainset_df
    gc.collect()


def f_conversion_ratio():
    """
    为 hdf_datatset 中的特征构造 conversion_ratio 特征，并存储到硬盘。
    """

    # 加载数据集
    trainset_df = pd.read_hdf(path_intermediate_dataset + hdf_trainset)

    # 安全的将 hdf_numeric_features_set 清空, 以避免可能的重复添加
    # 注意与其他也使用 hdf_numeric_features_set 的代码的先后顺序！！！
    util.safe_remove(path_intermediate_dataset + hdf_numeric_features_set)
    # 遍历数据集中的有效特征
    for c in trainset_df.columns:
        if c not in (columns_set_without_count_ratio | columns_set_mismatch):
            util.f_conversion_ratio(trainset_df, c)

    del trainset_df
    gc.collect()


def fg_dataset(hdf_out, hdf_in):
    """
    为 trainset 和 testset_ol 添加已经构造好的特征。
    :param hdf_out: 
    :param hdf_in: 
    :return: 
    """

    # out_file = path_feature + hdf_out
    # if util.is_exist(out_file):
    #     return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_out)

    # 加载 hdf_in
    dataset_df = pd.read_hdf(path_intermediate_dataset + hdf_in)

    # # 为每个有效特征添加对应的 count_ratio 特征
    # for c in dataset_df.columns:
    #     if c not in (columns_set_without_count_ratio | columns_set_mismatch):
    #         in_file = path_feature + 'f_count_ratio_' + c + '.h5'
    #         count_ratio = pd.read_hdf(in_file)
    #         dataset_df = dataset_df.merge(count_ratio, how='left', on=c)
    #         del count_ratio
    #         gc.collect()

    # 为每个有效特征添加对应的 conversion_ratio 特征
    for c in dataset_df.columns:
        if c not in (columns_set_without_count_ratio | columns_set_mismatch | columns_set_discarded):
            in_file = path_feature + 'f_conversion_ratio_' + c + '.h5'
            conversion_ratio = pd.read_hdf(in_file)
            dataset_df = dataset_df.merge(conversion_ratio, how='left', on=c)
            # 填充缺失值
            dataset_df['conversion_ratio_' + c].fillna(0, inplace=True)
            del conversion_ratio
            gc.collect()

    # 将地理位置调整到省级
    # 放在这里的原因是：在构造转化率特征时，使用市级的地理位置信息而不是省级的。
    # dataset_df['hometown'] = (dataset_df['hometown'] / 100).astype(int)
    # dataset_df['residence'] = (dataset_df['residence'] / 100).astype(int)
    # 去掉 hometown
    del dataset_df['hometown']

    # 构造是否 is_wifi 特征
    dataset_df[fn_is_wifi] = dataset_df['connectionType'] == 1

    # 为了在构造转化率特征时，使用完整的年龄信息而不是年龄段信息。所以到这里才给age分段
    age_interval = [0, 1, 4, 14, 29, 44, 59, 74, 84]
    dataset_df['age'] = pd.cut(dataset_df['age'], age_interval, right=False, include_lowest=True, labels=False)

    # # 加载并添加用户的活跃度特征
    # util.print_constructing_feature(fn_user_activity)
    # dataset_df = util.add_feature(dataset_df, hdf_user_activity, f_user_activity)
    # # 将 user_activity 的 NaN 填充为 5
    # dataset_df[fn_user_activity].fillna(5, inplace=True)

    # 加载并添加用户对app的品类偏好特征
    dataset_df = util.add_feature(dataset_df, hdf_user_pref_cat, f_user_pref_cat)
    # 将 cat_pref 的 NaN 填充为 0
    dataset_df[fn_cat_pref].fillna(0, inplace=True)
    # 同时构造 is_pref_cat 特征
    dataset_df[fn_is_pref_cat] = dataset_df['appCategory'] == dataset_df[fn_cat_pref]

    # 添加 app 的流行度特征
    util.print_constructing_feature(fn_app_popularity)
    dataset_df = util.add_feature(dataset_df, hdf_app_popularity, f_app_popularity)
    # 将 app_popularity 离散化
    dataset_df[fn_app_popularity] = \
        pd.cut(dataset_df[fn_app_popularity], np.logspace(0, 7, num=8), include_lowest=True, labels=False)
    # 将 app_popularity 的 NaN 填充为 6
    dataset_df[fn_app_popularity].fillna(6, inplace=True)

    # 添加二次组合特征 user(age, gender, education, residence)-connectionType
    dataset_df[fn_age_connectionType] = util.elegant_pairing(dataset_df['age'], dataset_df['connectionType'])
    dataset_df[fn_gender_connectionType] = util.elegant_pairing(dataset_df['gender'], dataset_df['connectionType'])
    dataset_df[fn_education_connectionType] = \
        util.elegant_pairing(dataset_df['education'], dataset_df['connectionType'])
    dataset_df[fn_marriageStatus_connectionType] = \
        util.elegant_pairing(dataset_df['marriageStatus'], dataset_df['connectionType'])
    dataset_df[fn_residence_connectionType] = \
        util.elegant_pairing(dataset_df['residence'], dataset_df['connectionType'])
    dataset_df[fn_appID_is_wifi] = \
        util.elegant_pairing(dataset_df['appID'], dataset_df[fn_is_wifi])

    # 添加二次组合特征 user-appCategory
    dataset_df[fn_age_appCategory] = util.elegant_pairing(dataset_df['age'], dataset_df['appCategory'])
    dataset_df[fn_gender_appCategory] = util.elegant_pairing(dataset_df['gender'], dataset_df['appCategory'])
    dataset_df[fn_education_appCategory] = util.elegant_pairing(dataset_df['education'], dataset_df['appCategory'])
    # dataset_df[fn_marriageStatus_appCategory] = \
    #     util.elegant_pairing(dataset_df['marriageStatus'], dataset_df['appCategory'])
    dataset_df[fn_haveBaby_appCategory] = util.elegant_pairing(dataset_df['haveBaby'], dataset_df['appCategory'])

    # 添加 connectionType-appCategory
    dataset_df[fn_appCategory_connectionType] = \
        util.elegant_pairing(dataset_df['connectionType'], dataset_df['appCategory'])

    # 添加“该 userID_appID 是否已存在安装行为”的特征
    # 从 action 数据构造
    if 'train' in hdf_in:
        dataset_df = f_is_installed_from_action_trainset(dataset_df)
    elif 'test' in hdf_in:
        dataset_df = f_is_installed_from_action_testset_ol(dataset_df)
    # 从 user_app 数据构造
    # dataset_df['userID-appID'] = util.elegant_pairing(dataset_df['userID'], dataset_df['appID'])
    userID_appID = pd.read_hdf(path_intermediate_dataset + hdf_userID_appID_pair_installed)
    dataset_df[fn_is_installed] |= dataset_df['userID_appID'].isin(userID_appID)
    del dataset_df['userID_appID']
    del userID_appID
    gc.collect()

    # # 添加样本的置信度特征
    # if 'train' in hdf_in:
    #     dataset_df = f_confidence_trainset(trainset_df=dataset_df)
    # elif 'test' in hdf_in:
    #     dataset_df = f_confidence_testset_ol(testset_ol=dataset_df)

    # if 'train' in hdf_in:
    #     # 舍弃后一个小时的样本
    #     dataset_df = dataset_df.loc[(dataset_df['clickTime'] < 302300)]

    # 删除不匹配的列
    for c in columns_set_mismatch | columns_set_inapparent:
        if c in dataset_df.columns:
            del dataset_df[c]

    # 准备存储，删除它们以避免干扰 one-hot    
    del dataset_df['clickTime']
    if 'train' in hdf_in:
        del dataset_df['conversionTime']
    elif 'test' in hdf_in:
        del dataset_df['label']
        del dataset_df['instanceID']

    # 检查缺失值
    if util.check_null(dataset_df):
        gc.collect()
    else:
        print('通过缺失值检查，不存在缺失值。')

    # 存储
    util.safe_save(path_feature, hdf_out, dataset_df)

    # # 打印每个特征的样本比例
    # if 'train' in hdf_in:
    #     util.print_sample_ratio(dataset_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def fg_trainset():
    """
    为 trainset 添加已经构造好的特征。
    :return: 
    """
    fg_dataset(hdf_trainset_fg, hdf_trainset)


def fg_testset_ol():
    """
    为 testset_ol 添加已经构造好的特征。
    :return: 
    """
    fg_dataset(hdf_testset_ol_fg, hdf_testset_ol)


def merge(hdf_out, hdf_in):
    # 开始计时，并打印相关信息
    start = util.print_start(hdf_out)

    # 加载数据
    user = pd.read_hdf(path_intermediate_dataset + hdf_user_fg)
    context = pd.read_hdf(path_intermediate_dataset + hdf_in)

    # 合并为 dataset
    dataset = pd.merge(context, user, on='userID')
    del user
    del context
    gc.collect()

    ad = pd.read_hdf(path_intermediate_dataset + hdf_ad_fg)
    dataset = dataset.merge(ad, on='creativeID')
    del ad
    gc.collect()

    # 构造 is_pref_cat 特征
    dataset[fn_is_pref_cat] = dataset['appCategory'] == dataset[fn_cat_pref]

    # 删除 creativeID, userID 两列
    del dataset['creativeID']
    del dataset['userID']

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_out, dataset)

    # 停止计时，并打印相关信息
    util.print_stop(start)


# 这两个是每一次迭代特征都需要重新生成
def merge_dataset():
    merge(hdf_trainset_fg, hdf_context_dataset_fg)


def merge_testset_ol():
    merge(hdf_testset_ol_fg, hdf_context_testset_ol_fg)


def construct_feature():
    # 计时开始
    start = time()

    # fg_ad()
    # fg_user()
    # fg_context_dataset()
    # fg_context_testset_ol()
    # merge_dataset()
    # merge_testset_ol()
    # f_count_ratio()
    f_conversion_ratio()
    fg_trainset()
    fg_testset_ol()

    print('\nThe total time spent on constructing feature: {0:.2f} s'.format(time() - start))
