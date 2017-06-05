import contextlib
import gc
import os
from time import time
from pandas import DataFrame, Series
import pandas as pd

import numpy as np
from scipy.sparse import save_npz
from sklearn.externals import joblib

from predefine import *


def is_exist(file):
    if os.path.exists(file):
        print(file + ' existed.')
        return True
    return False


def safe_remove(filename):
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


def safe_save(path, file_name, obj):
    # 确保 path 存在，否则后面存储时会报错，这样就能在初次部署代码时自动创建目录了
    os.makedirs(path, exist_ok=True)

    out_file = path + file_name
    safe_remove(out_file)
    num_dot = file_name.count('.')
    if num_dot != 1:
        print("文件名 {0} 有误: '.'的数量超过1个。".format(file_name))
        return
    # 获取不带后缀的文件名，前提是file_name中只存在一个'.'
    name = file_name.split('.')[0]
    # 获取文件后缀名
    suffix = file_name.split('.')[-1]
    # 存储
    if suffix == 'h5':
        obj.to_hdf(out_file, key=name, mode='w')
    elif suffix == 'npz':
        save_npz(out_file, obj)
    elif suffix == 'npy':
        np.save(out_file, obj)
    elif suffix == 'pkl':
        joblib.dump(obj, out_file)
    elif suffix == 'csv':
        obj.to_csv(out_file)
        import zipfile
        zip_file = zipfile.ZipFile(path + name + '.zip', 'w')
        zip_file.write(
            out_file,
            arcname='submission.csv',
            compress_type=zipfile.ZIP_DEFLATED
        )
    # 手动释放内存
    del obj
    gc.collect()


def print_start(file_name):
    start = time()
    print('\nCalculating ' + file_name + '...')
    return start


def print_stop(start):
    print('Completed. Time used = {0:.2f} s'.format(time() - start))


def elegant_pairing(s1, s2):
    """
    寡人原创的并行化实现。原理见：http://szudzik.com/ElegantPairing.pdf
    :param s1: Series
    :param s2: Series
    :return: 
    """
    arr1 = s1.values
    arr2 = s2.values
    flag = arr1 >= arr2
    res = flag * (arr1 * arr1 + arr1 + arr2) + (~flag) * (arr1 + arr2 * arr2)
    return res


def min_max_scaling(s):
    """
    用于在构造特征时进行归一化。
    :param s: pandas.Series 对象
    :return: 完成归一化的 pandas.Series 对象
    """
    mx = s.max()
    mn = s.min()
    s = (s - mn) / (mx - mn)
    return s


def f_count_ratio(df, column):
    """
    对 df 中的特定 column 构造 click_count, conversion_count, conversion_ratio 特征，并存储到硬盘。
    :param df: pandas.DataFrame 对象
    :param column: string, 特定 column
    
    Notes
    -----
    存储到硬盘而不是返回结果，是因为 context_train 和 context_test_ol 都需要这些特征，以方便两者做合并。
    """

    hdf_out = 'f_count_ratio_' + column + '.h5'

    click_count_column = 'click_count_' + column
    conversion_count_column = 'conversion_count_' + column
    conversion_ratio_column = 'conversion_ratio_' + column

    # if is_exist(out_file):
    #     return

    # 开始计时，并打印相关信息
    start = print_start(hdf_out)

    count = DataFrame(df[column].value_counts())
    count.reset_index(inplace=True)
    count.columns = [column, click_count_column]

    conversion_count = DataFrame(df.loc[df['label'] == 1, column].value_counts())
    conversion_count.reset_index(inplace=True)
    conversion_count.columns = [column, conversion_count_column]

    count_ratio = count.merge(conversion_count, how='left', on=column)
    count_ratio[conversion_count_column].fillna(0, inplace=True)

    del count
    del conversion_count
    gc.collect()

    count_ratio[conversion_ratio_column] = count_ratio[conversion_count_column] / count_ratio[click_count_column]

    # 对 click_count_column 分组, 目前的程序应该用不上这段代码，因为userID被舍弃了
    if column == 'userID':
        bins = [1, 28, 44, 120]
        count_ratio[click_count_column] = \
            pd.cut(count_ratio[click_count_column], bins=bins, include_lowest=True, labels=False)

    # 取对数，归一化。无论是类别特征还是数值特征，都可以做这样的处理，没有坏处。
    for c in count_ratio.columns.values[1:]:
        # 错在这里（2017-5-26 19:10:00）
        indexer = count_ratio[c] != 0
        count_ratio.loc[indexer, c] = np.log10(count_ratio.loc[indexer, c])
        count_ratio[c] = min_max_scaling(count_ratio[c])

    # 之前是把这部分作为数值特征的，现在把他们删除掉试试
    if column in dense_feature_name_set:
        del count_ratio[click_count_column]

    # 存储
    safe_save(path_feature, hdf_out, count_ratio)

    # 停止计时，并打印相关信息
    print_stop(start)


def f_conversion_ratio(df, column):
    """
    对 df 中的特定 column 构造 conversion_ratio 特征，并存储到硬盘。
    :param df: pandas.DataFrame 对象
    :param column: string, 特定 column

    Notes
    -----
    存储到硬盘而不是返回结果，是因为 context_train 和 context_test_ol 都需要这些特征，以方便两者做合并。
    """

    hdf_out = 'f_conversion_ratio_' + column + '.h5'

    click_count_column = 'click_count_' + column
    conversion_count_column = 'conversion_count_' + column
    conversion_ratio_column = 'conversion_ratio_' + column

    # if is_exist(out_file):
    #     return

    # 开始计时，并打印相关信息
    start = print_start(hdf_out)

    count = DataFrame(df[column].value_counts())
    count.reset_index(inplace=True)
    count.columns = [column, click_count_column]

    conversion_count = DataFrame(df.loc[df['label'] == 1, column].value_counts())
    conversion_count.reset_index(inplace=True)
    conversion_count.columns = [column, conversion_count_column]

    count_ratio = count.merge(conversion_count, how='left', on=column)
    count_ratio[conversion_count_column].fillna(0, inplace=True)

    del count
    del conversion_count
    gc.collect()

    count_ratio[conversion_ratio_column] = count_ratio[conversion_count_column] / count_ratio[click_count_column]
    del count_ratio[conversion_count_column]
    del count_ratio[click_count_column]

    # conversion_ratio 没有必要取对数，但是要归一化
    count_ratio[conversion_ratio_column] = min_max_scaling(count_ratio[conversion_ratio_column])

    # 存储
    safe_save(path_feature, hdf_out, count_ratio)

    # 停止计时，并打印相关信息
    print_stop(start)


def f_click_count(df, column):
    """
    对 df 中的特定 column 构造 click_count 特征，并存储到硬盘。
    :param df: pandas.DataFrame 对象
    :param column: string, 特定 column

    Notes
    -----
    存储到硬盘而不是返回结果，是因为 context_train 和 context_test_ol 都需要这些特征，以方便两者做合并。
    """

    hdf_out = 'f_click_count_' + column + '.h5'

    click_count_column = 'click_count_' + column
    click_count_square_column = 'click_count_square_' + column

    # 开始计时，并打印相关信息
    start = print_start(hdf_out)

    count = DataFrame(df[column].value_counts())
    count.reset_index(inplace=True)
    count.columns = [column, click_count_column]

    # 归一化
    count[click_count_column] = min_max_scaling(count[click_count_column])
    count[click_count_square_column] = count[click_count_column] ** 2

    # 存储
    safe_save(path_feature, hdf_out, count)

    # 停止计时，并打印相关信息
    print_stop(start)


def f_conversion_count(df, column):
    """
    对 df 中的特定 column 构造 conversion_count 特征，并存储到硬盘。
    :param df: pandas.DataFrame 对象
    :param column: string, 特定 column

    Notes
    -----
    存储到硬盘而不是返回结果，是因为 context_train 和 context_test_ol 都需要这些特征，以方便两者做合并。
    """

    hdf_out = 'f_conversion_count_' + column + '.h5'

    # click_count_column = 'click_count_' + column
    conversion_count_column = 'conversion_count_' + column

    # 开始计时，并打印相关信息
    start = print_start(hdf_out)

    # count = DataFrame(df[column].value_counts())
    # count.reset_index(inplace=True)
    # count.columns = [column, click_count_column]

    conversion_count = DataFrame(df.loc[df['label'] == 1, column].value_counts())
    conversion_count.reset_index(inplace=True)
    conversion_count.columns = [column, conversion_count_column]

    # del count
    # gc.collect()

    # 存储
    safe_save(path_feature, hdf_out, conversion_count)

    # 停止计时，并打印相关信息
    print_stop(start)


def exist_null(df):
    """
    检查一个 DataFrame 中是否有 null.
    :param df: 
    :return: 
    """

    print('\nChecking missing value...')
    res = False
    for c in df.columns:
        size = df.loc[df[c].isnull()].index.size
        if size != 0:
            res = True
            print('There is {0} missing values in {1}.'.format(size, c))
    return res


def check_match():
    """
    检查 hdf_trainset_fg, hdf_testset_ol_fg 中类别特征所对应的列的取值是否匹配。
    """

    # 开始计时，并打印相关信息
    start = time()
    print('\n开始检查 hdf_trainset_fg, hdf_testset_ol_fg 中类别特征所对应的列的取值是否匹配')

    trainset_df = pd.read_hdf(path_feature + hdf_trainset_fg)
    testset_ol_df = pd.read_hdf(path_feature + hdf_testset_ol_fg)

    # 删除 'label' 列，再做后续比较。
    del trainset_df['label']

    columns_set_trainset = set(trainset_df.columns)
    columns_set_testset_ol = set(testset_ol_df.columns)
    if columns_set_trainset != columns_set_testset_ol:
        return False
    flag = True
    # 检查每一列的取值是否匹配
    for c in (columns_set_trainset - numeric_features_set):
        st1 = set(trainset_df[c])
        st2 = set(testset_ol_df[c])
        if st1 != st2:
            flag = False
            print('{} 不匹配。'.format(c))
    if flag:
        print('完全匹配')

    # 停止计时，并打印相关信息
    print_stop(start)


def print_value_count(df, name):
    """
    打印类别特征取值个数。
    """

    # 开始计时，并打印相关信息
    start = time()
    print('\n开始检查 {0} 中类别特征取值个数……'.format(name))

    d = {}
    for column in (set(df.columns) - numeric_features_set):
        d[column] = df[column].value_counts().index.size
    s = Series(d).sort_values()
    print(s)
    print('sum:{0}'.format(s.sum()))

    # 停止计时，并打印相关信息
    print_stop(start)


def check_value_count():
    """
    检查 hdf_trainset_fg, hdf_testset_ol_fg 中类别特征的取值个数。
    """

    trainset_df = pd.read_hdf(path_feature + hdf_trainset_fg)
    print_value_count(trainset_df, hdf_trainset_fg)
    del trainset_df
    gc.collect()

    testset_ol_df = pd.read_hdf(path_feature + hdf_testset_ol_fg)
    print_value_count(testset_ol_df, hdf_testset_ol_fg)
    del testset_ol_df
    gc.collect()


def add_feature(df, hdf_file, gen_func):
    """
    为数据集添加特征。
    :param df: 数据集。
    :param hdf_file: 特征文件。
    :param gen_func: 生成该特征的函数
    :return: 
    """

    # 加载并添加特征
    in_file = path_feature + hdf_file
    if not os.path.exists(in_file):
        gen_func()
    feature = pd.read_hdf(in_file)

    # 成立的条件是对应的合并column处于第1列
    key = feature.columns.values[0]
    for fn_feature in feature.columns.values[1:]:
        # 打印进度信息
        print_constructing_feature(fn_feature)

    # 添加特征
    df = df.merge(feature, how='left', on=key)

    # 手动释放内存
    del feature
    gc.collect()

    return df


def to_minute(df, column):
    """
    将官方编码的时间转换为以分钟表示的时间。
    :param df:
    :param column: 时间列。
    """
    indexer = ~df[column].isnull()
    day = np.floor(df.loc[indexer, column] / 10000)
    hour = np.floor(df.loc[indexer, column] / 100) % 100
    minute = df.loc[indexer, column] % 100
    df.loc[indexer, column + '_min'] = day * 24 * 60 + hour * 60 + minute


def print_constructing_feature(fn_feature):
    """
    打印正在构造的特征的信息。
    :return:
    """
    print('    {0}...'.format(fn_feature))


def print_all_column_sample_ratio(df):
    """
    打印负正样本比例。
    :param df:
    :return:
    """

    for c in df.columns:
        if c not in columns_set_without_count_ratio:
            d = {}
            key_set = set(df[c].values)
            for key in key_set:
                sample_num_negative = df.loc[(df['label'] != 1) & (df[c] == key)].index.size
                sample_num_positive = df.loc[(df['label'] == 1) & (df[c] == key)].index.size
                if sample_num_positive != 0:
                    sample_ratio = np.round(sample_num_negative / sample_num_positive, 4)
                else:
                    sample_ratio = np.nan
                d[c + '_' + str(key)] = sample_ratio
            res = DataFrame(d, index=['sample_ratio'])
            print(res)
            # print("sample ratio of '{0}': {1}".format(c, int(sample_num_negative / sample_num_positive)))


def print_all_column_sample_ratio_min_max():
    """
    打印负正样本比例。
    """

    start = time()

    df = pd.read_hdf(path_feature + hdf_trainset_fg)

    import csv
    with open(path_intermediate_dataset + 'sample_ratio.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for c in df.columns:
            if 'conversion_ratio' in c:
                continue
            mn = 2 ** 31
            mx = -2 ** 31
            key_set = set(df[c].values)
            for key in key_set:
                indexer_key = df[c] == key
                indexer_positive = df['label'] == 1

                sample_num_negative = df.loc[(~indexer_positive) & indexer_key].index.size
                sample_num_positive = df.loc[indexer_positive & indexer_key].index.size
                if sample_num_positive != 0:
                    sample_ratio = np.round(sample_num_negative / sample_num_positive, 4)
                else:
                    sample_ratio = np.nan

                if sample_ratio < mn:
                    mn = sample_ratio
                if sample_ratio > mx:
                    mx = sample_ratio
            writer.writerow(['min', str(mn), 'max', str(mx), c])
            print('\nsample ratio of {0}'.format(c))
            print('min: {0}, max: {1}'.format(mn, mx))
    print_stop(start)


def get_sample_ratio(df, c):
    """
    获取特征的每个取值负正样本比例。
    :param c:
    :param df:
    :return:
    """

    d = {}
    key_set = set(df[c].values)
    for key in key_set:
        indexer_key = df[c] == key
        sample_num_negative = df.loc[(df['label'] != 1) & indexer_key].index.size
        sample_num_positive = df.loc[(df['label'] == 1) & indexer_key].index.size
        if sample_num_positive != 0:
            sample_ratio = int(sample_num_negative / sample_num_positive)
        else:
            sample_ratio = np.nan
        d[key] = sample_ratio
    return Series(d).sort_values(ascending=False)


def get_new_cat_list(sample_ratio, rg):
    """
    获取一系列分类的 set
    :param sample_ratio:
    :param rg:
    :return:
    """
    new_cats_list = [set(sample_ratio.loc[sample_ratio.isnull()].index)]
    value_set_0_begin = set(sample_ratio.loc[(sample_ratio >= 0) & (sample_ratio <= rg[0])].index)
    value_set_end_infinity = set(sample_ratio.loc[sample_ratio > rg[-1] + 10].index)
    for value in value_set_0_begin:
        new_cats_list.append({value})
    for begin in rg:
        end = begin + 9
        new_cats_list.append(set(sample_ratio.loc[(sample_ratio >= begin) & (sample_ratio <= end)].index))
    for value in value_set_end_infinity:
        new_cats_list.append({value})
    return new_cats_list


def assign_new_cat(df, column, new_cat_list):
    """
    分配新类。
    :param df:
    :param column:
    :param new_cat_list:
    :return:
    """

    new_column = column + '_cat'
    for cat_label in range(len(new_cat_list)):
        indexer = df[column].isin(new_cat_list[cat_label])
        df.loc[indexer, new_column] = cat_label
    return df


def add_combi_feature(df):
    """
    根据二次组合特征列表添加组合特征。
    :param df:
    :return:
    """

    for f1, f2 in combi_feature:
        f = f1 + '_' + f2
        print_constructing_feature(f)
        df[f] = elegant_pairing(df[f1], df[f2])
    return df


def f_count_ratio_for_scoring(df, column):
    neg_count_column = 'neg_count_' + column
    pos_count_column = 'pos_count_' + column
    sample_ratio_column = 'sample_ratio_' + column

    count = DataFrame(df.loc[df['label'] == 0, column].value_counts())
    count.reset_index(inplace=True)
    count.columns = [column, neg_count_column]

    pos_count = DataFrame(df.loc[df['label'] == 1, column].value_counts())
    pos_count.reset_index(inplace=True)
    pos_count.columns = [column, pos_count_column]

    count_ratio = count.merge(pos_count, how='left', on=column)
    count_ratio[pos_count_column].fillna(0, inplace=True)

    del count
    del pos_count
    gc.collect()

    count_ratio[sample_ratio_column] = count_ratio[neg_count_column] / count_ratio[pos_count_column]

    return count_ratio


def get_feature_pos_score(df, c):
    """
    获得特征辨认正样本的评分。
    :param df:
    :param c:
    :return:
    """
    pos_count_column = 'pos_count_' + c
    sample_ratio_column = 'sample_ratio_' + c

    count_ratio = f_count_ratio_for_scoring(df, c)
    count_ratio.sort_values(by=[sample_ratio_column])
    score = count_ratio.loc[count_ratio[sample_ratio_column] < 10, pos_count_column].sum()
    del count_ratio
    gc.collect()
    return score


def get_feature_pos_score_series(df, columns_no_traverse):
    """
    获得一系列特征辨认正样本的评分。
    :param df:
    :param columns_no_traverse:
    :return:
    """
    score_dict = {}
    for c in df.columns:
        if c in columns_no_traverse:
            continue
        score = get_feature_pos_score(df, c)
        score_dict[c] = score
    score_series = Series(score_dict)
    score_series.sort_values(ascending=False, inplace=True)
    return score_series


def analyze_trainset_feature():
    """
    分析trainset中的特征。
    :return:
    """

    trainset_fg_df = pd.read_hdf(path_feature + hdf_trainset_fg)

    columns_no_traverse = {
        'label',
        'clickTime',
        'conversionTime',
        'instanceID',
        'userID'
    }
    pos_score_series = get_feature_pos_score_series(trainset_fg_df, columns_no_traverse)
    print(pos_score_series)

    # 存储
    safe_save(path_analysis_res, hdf_pos_score_series, pos_score_series)
    print('Has been stored to {0}.'.format(path_analysis_res + hdf_pos_score_series))
