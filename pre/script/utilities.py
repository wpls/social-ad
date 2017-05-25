import contextlib
import gc
import os
from time import time

import numpy as np
from scipy.sparse import save_npz
from sklearn.externals import joblib


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
        print("文件名有误: '.'的数量超过1个。")
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
