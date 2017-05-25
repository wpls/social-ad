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
    out_file = path_intermediate_dataset + hdf_app_popularity
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_app_popularity)

    # 加载 user_app
    user_app = pd.read_hdf(path_intermediate_dataset + hdf_user_app)

    # 提取 app 的热度特征
    app_popularity = user_app.groupby('appID').count()
    app_popularity.rename(columns={'userID': 'app_popularity'}, inplace=True)

    # 手动释放内存
    del user_app
    gc.collect()

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_app_popularity, app_popularity)

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
    ad['app_popularity'] = pd.cut(ad['app_popularity'], np.logspace(0, 7, num=8), include_lowest=True, labels=False)

    # 将 app_popularity 的 NaN 填充为 6
    ad['app_popularity'].fillna(6, inplace=True)

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
    out_file = path_intermediate_dataset + hdf_user_activity
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_user_activity)

    # 提取用户的活跃度特征
    user_app = pd.read_hdf(path_intermediate_dataset + hdf_user_app)
    user_activity = user_app.groupby('userID').count()
    user_activity.rename(columns={'appID': 'user_activity'}, inplace=True)

    # 手动释放内存
    del user_app
    gc.collect()

    # 离散化
    interval = np.ceil(np.logspace(0, 3, 6))
    user_activity['user_activity'] = \
        pd.cut(user_activity['user_activity'], interval, include_lowest=True, labels=False)
    user_activity.reset_index(inplace=True)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_user_activity, user_activity)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def f_user_pref_cat():
    out_file = path_intermediate_dataset + hdf_user_pref_cat
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
    user_pref_cat.rename(columns={'appCategory': 'cat_pref'}, inplace=True)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_user_pref_cat, user_pref_cat)

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

    # 对 age 分段
    age_interval = [0, 1, 4, 14, 29, 44, 59, 74, 84]
    user['age'] = pd.cut(user['age'], age_interval, right=False, include_lowest=True, labels=False)

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
    user['user_activity'].fillna(5, inplace=True)

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
    user['cat_pref'].fillna(0, inplace=True)

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


def f_hour():
    """
    构造与 hour 相关的特征。
    """
    out_file = path_intermediate_dataset + hdf_hour
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_hour)

    # 加载 train.h5
    train_df = pd.read_hdf(path_intermediate_dataset + hdf_train)

    # 从`clickTime`中提取`hour`特征
    train_df['hour'] = (train_df['clickTime'] / 100 % 100).astype(int)

    # 提取`hour`的`weight`特征
    hour_count_positive = train_df.loc[train_df['label'] == 1, 'hour'].value_counts()
    hour_count_positive.sort_index(inplace=True)

    hour_count_positive_df = DataFrame(hour_count_positive)
    hour_count_positive_df.reset_index(inplace=True)
    hour_count_positive_df.columns = ['hour', 'hour_weight']

    # 提取`hour`的`conversion_ratio`特征
    hour_count = train_df['hour'].value_counts()
    hour_count.sort_index(inplace=True)

    hour_count_df = DataFrame(hour_count)
    hour_count_df.reset_index(inplace=True)
    hour_count_df.columns = ['hour', 'hour_weight']

    hour_count_positive_df['hour_conversion_ratio'] = \
        hour_count_positive_df['hour_weight'] / hour_count_df['hour_weight']

    # hour_conversion_ratio 归一化
    mx = hour_count_positive_df['hour_conversion_ratio'].max()
    mn = hour_count_positive_df['hour_conversion_ratio'].min()
    hour_count_positive_df['hour_conversion_ratio'] = \
        (hour_count_positive_df['hour_conversion_ratio'] - mn) / (mx - mn)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_hour, hour_count_positive_df)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def f_week():
    """
    构造与 week 相关的特征。
    """
    out_file = path_intermediate_dataset + hdf_week
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_hour)

    # 加载 train.h5
    train_df = pd.read_hdf(path_intermediate_dataset + hdf_train)

    # 从`clickTime`中提取`week`特征
    train_df['week'] = (train_df['clickTime'] / 10000).astype(int) % 7

    # 提取`week`的`weight`特征
    week_count_positive = train_df.loc[train_df['label'] == 1, 'week'].value_counts()
    week_count_positive.sort_index(inplace=True)

    week_count_positive_df = DataFrame(week_count_positive)
    week_count_positive_df.reset_index(inplace=True)
    week_count_positive_df.columns = ['week', 'week_weight']

    # 提取`week`的`conversion_ratio`特征
    week_count = train_df['week'].value_counts()
    week_count.sort_index(inplace=True)

    week_count_df = DataFrame(week_count)
    week_count_df.reset_index(inplace=True)
    week_count_df.columns = ['week', 'week_weight']

    week_count_positive_df['week_conversion_ratio'] = \
        week_count_positive_df['week_weight'] / week_count_df['week_weight']

    # hour_conversion_ratio 归一化
    mx = week_count_positive_df['week_conversion_ratio'].max()
    mn = week_count_positive_df['week_conversion_ratio'].min()
    week_count_positive_df['week_conversion_ratio'] = \
        (week_count_positive_df['week_conversion_ratio'] - mn) / (mx - mn)

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_week, week_count_positive_df)

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


def f_conversion_ratio_connectionType():
    out_file = path_intermediate_dataset + hdf_conversion_ratio_connectionType
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_conversion_ratio_connectionType)

    # 加载数据
    train_df = pd.read_hdf(path_intermediate_dataset + hdf_train)
    # 提取 connectionType 的转化率特征
    distribution_connectionType = train_df['connectionType'].value_counts(dropna=False)
    distribution_connectionType.sort_index(inplace=True)

    distribution_connectionType_positive = train_df.loc[train_df['label'] == 1, 'connectionType'].value_counts(
        dropna=False)
    distribution_connectionType_positive.sort_index(inplace=True)

    # 手动释放内存
    del train_df
    gc.collect()

    distribution_connectionType_ratio = distribution_connectionType_positive / distribution_connectionType

    mx = distribution_connectionType_ratio.max()
    mn = distribution_connectionType_ratio.min()
    distribution_connectionType_ratio = (distribution_connectionType_ratio - mn) / (mx - mn)

    conversion_ratio_connectionType = DataFrame(distribution_connectionType_ratio)
    conversion_ratio_connectionType.reset_index(inplace=True)
    conversion_ratio_connectionType.columns = ['connectionType', 'conversion_ratio_connectionType']

    # 手动释放内存
    del distribution_connectionType_ratio
    gc.collect()

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_conversion_ratio_connectionType, conversion_ratio_connectionType)

    # 停止计时，并打印相关信息
    util.print_stop(start)


def f_conversion_ratio_telecomsOperator():
    out_file = path_intermediate_dataset + hdf_conversion_ratio_telecomsOperator
    if util.is_exist(out_file):
        return

    # 开始计时，并打印相关信息
    start = util.print_start(hdf_conversion_ratio_telecomsOperator)

    # 加载数据
    train_df = pd.read_hdf(path_intermediate_dataset + hdf_train)
    # 提取 telecomsOperator 的转化率特征
    distribution_telecomsOperator = train_df['telecomsOperator'].value_counts(dropna=False)
    distribution_telecomsOperator.sort_index(inplace=True)

    distribution_telecomsOperator_positive = train_df.loc[train_df['label'] == 1, 'telecomsOperator'].value_counts(
        dropna=False)
    distribution_telecomsOperator_positive.sort_index(inplace=True)

    # 手动释放内存
    del train_df
    gc.collect()

    distribution_telecomsOperator_ratio = distribution_telecomsOperator_positive / distribution_telecomsOperator

    mx = distribution_telecomsOperator_ratio.max()
    mn = distribution_telecomsOperator_ratio.min()
    distribution_telecomsOperator_ratio = (distribution_telecomsOperator_ratio - mn) / (mx - mn)

    conversion_ratio_telecomsOperator = DataFrame(distribution_telecomsOperator_ratio)
    conversion_ratio_telecomsOperator.reset_index(inplace=True)
    conversion_ratio_telecomsOperator.columns = ['telecomsOperator', 'conversion_ratio_telecomsOperator']

    # 手动释放内存
    del distribution_telecomsOperator_ratio
    gc.collect()

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_conversion_ratio_telecomsOperator, conversion_ratio_telecomsOperator)

    # 停止计时，并打印相关信息
    util.print_stop(start)


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
        f_conversion_ratio_connectionType()
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
    df['is_installed'] = df['userID-appID'].isin(userID_appID)

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


# 以上的是每一次迭代特征时，有改动的部分需要重新生成，可以通过提前手动
# 删除来实现，否则如果已经存在，则跳过计算


# ========== feature construction ==========

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
    dataset['is_pref_cat'] = dataset['appCategory'] == dataset['cat_pref']

    # 删除 creativeID, userID 两列
    del dataset['creativeID']
    del dataset['userID']

    # 存储
    util.safe_save(path_intermediate_dataset, hdf_out, dataset)

    # 停止计时，并打印相关信息
    util.print_stop(start)


# 这两个是每一次迭代特征都需要重新生成
def merge_dataset():
    merge(hdf_dataset_fg, hdf_context_dataset_fg)


def merge_testset_ol():
    merge(hdf_testset_ol_fg, hdf_context_testset_ol_fg)


def construct_feature():
    # 计时开始
    start = time()

    fg_ad()
    fg_user()
    fg_context_dataset()
    fg_context_testset_ol()
    merge_dataset()
    merge_testset_ol()

    print('\nThe total time spent on constructing feature: {0:.0f} s'.format(time() - start))
