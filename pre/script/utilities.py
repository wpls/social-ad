import contextlib
import gc
import os
from time import time
from pandas import DataFrame

import numpy as np
from scipy.sparse import save_npz
from sklearn.externals import joblib

from predefine import *


def is_exist(file):
    if os.path.exists(file):
        print('\n' + file + ' 已存在')
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
            arcname=file_name,
            compress_type=zipfile.ZIP_DEFLATED
        )
    # 手动释放内存
    del obj
    gc.collect()


def print_start(file_name):
    start = time()
    print('\nStart calculating ' + file_name + ' ……')
    return start


def print_stop(start):
    print('The calculation is complete.')
    print('time used = {0:.0f} s'.format(time() - start))


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


def f_count_ratio(df, column, numeric_features):
    """
    对 df 中的特定 column 构造 click_count, conversion_count, conversion_ratio 特征，并存储到硬盘。
    :param numeric_features: 
    :param df: pandas.DataFrame 对象
    :param column: string, 特定 column
    :return: n * 4 的 DataFrame，第一列是column, 后三列是上述特征
    
    Notes
    -----
    存储到硬盘而不是返回结果，是因为 context_train 和 context_test_ol 都需要这些特征，以方便两者做合并。
    """

    hdf_out = 'f_count_ratio_' + column + '.h5'
    out_file = path_feature + hdf_out

    click_count_column = 'click_count_' + column
    conversion_count_column = 'conversion_count_' + column
    conversion_ratio_column = 'conversion_ratio_' + column

    # 区分数值特征与类别特征，方便后面做 one-hot 处理
    if column in dense_feature_name_set:
        if column == 'userID':
            numeric_features |= {conversion_count_column, conversion_ratio_column}
        else:
            numeric_features |= {click_count_column, conversion_count_column, conversion_ratio_column}
    else:
        numeric_features |= {conversion_count_column, conversion_ratio_column}

    if is_exist(out_file):
        return

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

    # 取对数，归一化。无论是类别特征还是数值特征，都可以做这样的处理，没有坏处。
    for c in count_ratio.columns:
        count_ratio[c] = np.log10(count_ratio.loc[count_ratio[c] != 0, c])
        count_ratio[c] = min_max_scaling(count_ratio[c])

    # 存储
    safe_save(path_feature, hdf_out, count_ratio)

    # 停止计时，并打印相关信息
    print_stop(start)
