import gc

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz, csc_matrix

# 缩写
# fe: Feature Engineering
# fc: feature construction
# f: feature
# fg: feature group

# 路径
path_pre = '../'
path_original_dataset = path_pre + 'original-dataset/'
path_intermediate_dataset = path_pre + 'intermediate-dataset/'
path_modeling_dataset = path_pre + 'modeling-dataset/'
path_model = path_pre + 'model/'
path_submission_dataset = path_pre + 'submission-dataset/'

# 原始文件的 hdf 存储（已经过清洗与裁剪以节省计算性能）
hdf_ad = 'ad.h5'
hdf_app_cat = 'app_cat.h5'
hdf_pos = 'pos.h5'
hdf_test = 'test.h5'
hdf_train = 'train.h5'
hdf_action = 'action.h5'
hdf_user = 'user.h5'
hdf_user_app = 'user_app.h5'

# 计算的中间结果（以此减少内存占用）
hdf_user_app_cat = 'user_app_cat.h5'
hdf_userID_appID = 'userID_appID.h5'
hdf_userID_appID_for_test = 'userID_appID_for_test.h5'

# 从单个原始特征中提取出的特征
hdf_user_pref_cat = 'f_user_pref_cat.h5'
hdf_user_cat_weight = 'f_user_cat_weight.h5'
hdf_app_popularity = 'f_app_popularity.h5'
hdf_user_activity = 'f_user_activity.h5'
hdf_hour_weight = 'f_hour_weight.h5'
hdf_conversion_ratio_connectionType = 'f_conversion_ratio_connectionType.h5'
hdf_conversion_ratio_telecomsOperator = 'f_conversion_ratio_telecomsOperator.h5'
hdf_clickTime = 'f_clickTime.h5'
hdf_userID = 'f_userID.h5'

# 特征群文件
hdf_action_context_fg = 'fg_action_context.h5'
hdf_action_context_test_ol_fg = 'fg_action_context_test_ol.h5'
hdf_ad_fg = 'fg_ad.h5'
hdf_user_fg = 'fg_user.h5'

# 最终的数据集文件
hdf_dataset = 'dataset.h5'
hdf_testset_ol = 'testset_ol.h5'

# 稀疏矩阵, 一次项
npz_ad = 'ad_csc.npz'
npz_ad_test_ol = 'ad_csc_test_ol.npz'
npz_context = 'context_csc.npz'
npz_context_test_ol = 'context_csc_test_ol.npz'
npz_user = 'user_csc.npz'
npz_user_test_ol = 'user_csc_test_ol.npz'
# 二次项
npz_user_ad = 'user_ad.npz'
npz_user_ad_test_ol = 'user_ad_test_ol.npz'
npz_user_context = 'user_context.npz'
npz_user_context_test_ol = 'user_context_test_ol.npz'
npz_ad_context = 'ad_context.npz'
npz_ad_context_test_ol = 'ad_context_test_ol.npz'

# ndarray
npy_y = 'y.npy'
npy_y_train = 'y_train.npy'
npy_y_test = 'y_test.npy'

# 稀疏矩阵
npz_X_linear = 'X_linear.npz'
npz_X_interactive = 'X_interactive.npz'
npz_X = 'X.npz'
npz_X_test_ol_linear = 'X_test_ol_linear.npz'
npz_X_test_ol_interactive = 'X_test_ol_interactive.npz'
npz_X_test_ol = 'X_test_ol.npz'

npz_X_train = 'X_train.npz'
npz_X_test = 'X_test.npz'


def concatenate_csc_matrices_by_columns(m1, m2):
    """ 自定义 csc_matrices 合并函数，以替代 hstack
    Notes
    -----
    原因是 hstack 在合并的时候占内存太大，并且效率不高。对于所有的稀疏矩阵，它都将
    其转化为统一的coo_matrix再做合并，然后再转化为指定格式。
    :param m1: csc_matrice
    :param m2: csc_matrice
    :return: 合并的结果
    """
    # 确保 m1, m2 为 csc_matrix，不然要出问题
    from scipy.sparse import isspmatrix_csc
    if not isspmatrix_csc(m1):
        m1 = m1.tocsc()
        gc.collect()
    if not isspmatrix_csc(m2):
        m2 = m2.tocsc()
        gc.collect()
    # 分解合成
    data = np.concatenate((m1.data, m2.data))
    indices = np.concatenate((m1.indices, m2.indices))
    indptr = m2.indptr + len(m1.data)
    indptr = indptr[1:]
    indptr = np.concatenate((m1.indptr, indptr))
    # 手动释放内存
    del m1
    del m2
    gc.collect()
    # 生成结果
    res = csc_matrix((data, indices, indptr))

    # 手动释放内存
    del data
    del indices
    del indptr
    gc.collect()

    return res


def gen_linear_term():
    from sklearn.preprocessing import OneHotEncoder

    context_features = \
        ['sitesetID', 'positionType', 'connectionType', 'telecomsOperator', 'hour', 'hour_weight', 'is_pref_cat']
    user_features = \
        ['age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'user_activity',
         'cat_pref']
    ad_features = \
        ['advertiserID', 'appPlatform', 'appCategory', 'app_popularity']

    # 加载 dataset
    dataset = pd.read_hdf(path_intermediate_dataset + hdf_dataset)

    # y
    y = dataset['label'].values
    # 存储
    np.save(path_modeling_dataset + npy_y, y)
    # 手动释放内存
    del y
    gc.collect()

    # one-hot context
    enc_context = OneHotEncoder()
    context_csc = enc_context.fit_transform(dataset[context_features].values).tocsc()
    # 存储
    save_npz(path_modeling_dataset + npz_context, context_csc)
    # 手动释放内存
    del context_csc
    gc.collect()

    # one-hot user
    enc_user = OneHotEncoder()
    user_csc = enc_user.fit_transform(dataset[user_features].values).tocsc()
    # 存储
    save_npz(path_modeling_dataset + npz_user, user_csc)
    # 手动释放内存
    del user_csc
    gc.collect()

    # one-hot ad
    enc_ad = OneHotEncoder()
    ad_csc = enc_ad.fit_transform(dataset[ad_features].values).tocsc()
    # 存储
    save_npz(path_modeling_dataset + npz_ad, ad_csc)
    # 手动释放内存
    del ad_csc
    gc.collect()

    # 释放 dataset
    del dataset
    gc.collect()

    # 加载 testset_ol
    testset_ol = pd.read_hdf(path_intermediate_dataset + hdf_testset_ol)

    # one-hot context
    context_csc_test_ol = enc_context.transform(testset_ol[context_features].values).tocsc()
    # 存储
    save_npz(path_modeling_dataset + npz_context_test_ol, context_csc_test_ol)
    # 手动释放内存
    del context_csc_test_ol
    gc.collect()

    # one-hot user
    user_csc_test_ol = enc_user.transform(testset_ol[user_features].values).tocsc()
    # 存储
    save_npz(path_modeling_dataset + npz_user_test_ol, user_csc_test_ol)
    # 手动释放内存
    del user_csc_test_ol
    gc.collect()

    # one-hot ad
    ad_csc_test_ol = enc_ad.transform(testset_ol[ad_features].values).tocsc()
    # 存储
    save_npz(path_modeling_dataset + npz_ad_test_ol, ad_csc_test_ol)
    # 手动释放内存
    del ad_csc_test_ol
    gc.collect()

    # 释放 testset_ol
    del testset_ol
    gc.collect()


def merge_X_linear():
    # 加载 ad, context
    ad = load_npz(path_modeling_dataset + npz_ad)
    context = load_npz(path_modeling_dataset + npz_context)
    # 合并
    # tmp = hstack([ad, context])
    tmp = concatenate_csc_matrices_by_columns(ad, context)
    # 手动释放内存
    del ad
    del context
    gc.collect()

    # 加载 user
    user = load_npz(path_modeling_dataset + npz_user)
    # linear = hstack([tmp, user])
    linear = concatenate_csc_matrices_by_columns(tmp, user)
    del tmp
    del user
    gc.collect()
    # 存储
    save_npz(path_modeling_dataset + npz_X_linear, linear)
    # 手动释放内存
    del linear
    gc.collect()


def merge_X_interactive():
    # 加载 interactive_term
    ad_context = load_npz(path_modeling_dataset + npz_ad_context)
    user_ad = load_npz(path_modeling_dataset + npz_user_ad)
    # 合并
    # tmp = hstack([ad_context, user_ad])
    tmp = concatenate_csc_matrices_by_columns(ad_context, user_ad)
    # 手动释放内存
    del ad_context
    del user_ad
    gc.collect()

    # 加载 interactive_term
    user_context = load_npz(path_modeling_dataset + npz_user_context)
    # 合并
    # interactive = hstack([tmp, user_context])
    interactive = concatenate_csc_matrices_by_columns(tmp, user_context)
    # 手动释放内存
    del tmp
    del user_context
    # 存储
    save_npz(path_modeling_dataset + npz_X_interactive, interactive)
    # 手动释放内存
    del interactive
    gc.collect()


# ========== interactive ==========
# 目前暂时不使用二次组合特征
def gen_interactive_term(in_file_1, in_file_2, out_file):
    # 注意传入参数的顺序，与效率有关，列数少的放前面

    # 加载 csc_matrix
    A = load_npz(path_modeling_dataset + in_file_1)
    B = load_npz(path_modeling_dataset + in_file_2)

    """  
    ===== old code =====
    # 生成 interactive_term
    A_B_list = []
    for i in range(np.shape(A)[1]):
        A_B_list.append(B.multiply(A[:, i]))
    res = hstack(A_B_list, format='csc')

    # 手动释放内存
    del A
    del B
    del A_B_list
    gc.collect()
    """

    # ===== new code =====
    res = B.multiply(A[:, 0]).tocsc()
    for i in range(1, np.shape(A)[1]):
        # 这里有个大问题，multiply 的返回结果是 csr
        product = B.multiply(A[:, i]).tocsc()
        gc.collect()
        res = concatenate_csc_matrices_by_columns(res, product)
        # 只保留元素不全为 0 的列
        res = res[:, res.getnnz(0) > 0]
        # 手动释放内存
        del product
        gc.collect()

    # 手动释放内存
    del A
    del B
    gc.collect()

    # 存储
    save_npz(path_modeling_dataset + out_file, res)

    # 手动释放 res
    del res
    gc.collect()


def gen_interactive_user_ad():
    gen_interactive_term(npz_ad, npz_user, npz_user_ad)


def gen_interactive_user_ad_test_ol():
    gen_interactive_term(npz_ad_test_ol, npz_user_test_ol, npz_user_ad_test_ol)


def gen_interactive_user_context():
    gen_interactive_term(npz_context, npz_user, npz_user_context)


def gen_interactive_user_context_test_ol():
    gen_interactive_term(npz_context_test_ol, npz_user_test_ol, npz_user_context_test_ol)


def gen_interactive_ad_context():
    gen_interactive_term(npz_ad, npz_context, npz_ad_context)


def gen_interactive_ad_context_test_ol():
    gen_interactive_term(npz_ad_test_ol, npz_context_test_ol, npz_ad_context_test_ol)


def gen_interactive_all():
    # user_ad
    gen_interactive_user_ad()
    gen_interactive_user_ad_test_ol()
    # user_context
    gen_interactive_user_context()
    gen_interactive_user_context_test_ol()
    # ad_context
    gen_interactive_ad_context()
    gen_interactive_ad_context_test_ol()


def merge_X():
    # 加载 linear_term
    linear = load_npz(path_modeling_dataset + npz_X_linear)
    # 加载 interactive_term
    interactive = load_npz(path_modeling_dataset + npz_X_interactive)

    # 合并出 X_csc
    X_csc = concatenate_csc_matrices_by_columns(linear, interactive)
    # 手动释放内存
    del linear
    del interactive
    gc.collect()

    # 转为 csr
    X = X_csc.tocsr()
    # 手动释放内存
    del X_csc
    gc.collect()
    # 存储
    save_npz(path_modeling_dataset + npz_X, X)
    # 手动释放内存
    del X
    gc.collect()


def merge_X_test_ol_linear():
    # 加载 ad, context
    ad = load_npz(path_modeling_dataset + npz_ad_test_ol)
    context = load_npz(path_modeling_dataset + npz_context_test_ol)
    # 合并
    # tmp = hstack([ad, context])
    tmp = concatenate_csc_matrices_by_columns(ad, context)
    # 手动释放内存
    del ad
    del context
    gc.collect()

    # 加载 user
    user = load_npz(path_modeling_dataset + npz_user_test_ol)
    # 合并
    # linear = hstack([tmp, user])
    linear = concatenate_csc_matrices_by_columns(tmp, user)
    # 手动释放内存
    del tmp
    del user
    gc.collect()
    # 存储
    save_npz(path_modeling_dataset + npz_X_test_ol_linear, linear)
    # 手动释放内存
    del linear
    gc.collect()


def merge_X_test_ol_interactive():
    # 加载 interactive_term
    ad_context = load_npz(path_modeling_dataset + npz_ad_context_test_ol)
    user_ad = load_npz(path_modeling_dataset + npz_user_ad_test_ol)
    # 合并
    # tmp = hstack([ad_context, user_ad])
    tmp = concatenate_csc_matrices_by_columns(ad_context, user_ad)
    # 手动释放内存
    del ad_context
    del user_ad
    gc.collect()

    # 加载 interactive_term
    user_context = load_npz(path_modeling_dataset + npz_user_context_test_ol)
    # 合并
    # interactive = hstack([tmp, user_context])
    interactive = concatenate_csc_matrices_by_columns(tmp, user_context)
    # 手动释放内存
    del tmp
    del user_context
    # 存储
    save_npz(path_modeling_dataset + npz_X_test_ol_interactive, interactive)
    # 手动释放内存
    del interactive
    gc.collect()


def merge_X_test_ol():
    # 加载 linear_term
    linear = load_npz(path_modeling_dataset + npz_X_test_ol_linear)
    # 加载 interactive_term
    interactive = load_npz(path_modeling_dataset + npz_X_test_ol_interactive)

    # 合并出 X_csc
    X_csc = concatenate_csc_matrices_by_columns(linear, interactive)
    # 手动释放内存
    del linear
    del interactive
    gc.collect()

    # 转为 csr
    X = X_csc.tocsr()
    # 手动释放内存
    del X_csc
    gc.collect()
    # 存储
    save_npz(path_modeling_dataset + npz_X_test_ol, X)
    # 手动释放内存
    del X
    gc.collect()

