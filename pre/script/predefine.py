# 路径
path_pre = '../'
path_original_dataset = path_pre + 'original-dataset/'
path_intermediate_dataset = path_pre + 'intermediate-dataset/'
path_feature = path_pre + 'feature/'
path_modeling_dataset = path_pre + 'modeling-dataset/'
path_model = path_pre + 'model/'
path_submission_dataset = path_pre + 'submission-dataset/'
path_cv_res = path_pre + 'cv-res/'
path_analysis_res = path_pre + 'analysis_res/'

# 获取比赛数据集的链接
url_original_dataset = 'http://spabigdata-1253211098.file.myqcloud.com/pre.zip'
zip_original_dataset = path_pre + 'pre.zip'

# 原始文件
csv_ad = 'ad.csv'
csv_app_cat = 'app_categories.csv'
csv_pos = 'position.csv'
csv_test = 'test.csv'
csv_train = 'train.csv'
csv_action = 'user_app_actions.csv'
csv_user = 'user.csv'
csv_user_app = 'user_installedapps.csv'

# 原始文件的 hdf 存储（已经过清洗与裁剪以节省计算性能）
hdf_ad = 'ad.h5'
hdf_app_cat = 'app_cat.h5'
hdf_pos = 'pos.h5'
hdf_test_ol = 'test_ol.h5'
hdf_train = 'train.h5'
hdf_action = 'action.h5'
hdf_user = 'user.h5'
hdf_user_app = 'user_app.h5'

# 计算的中间结果（以此减少内存占用）
hdf_valid = 'valid.h5'
hdf_user_app_cat = 'user_app_cat.h5'
hdf_userID_appID_pair_installed = 'userID_appID_pair_installed.h5'
hdf_trainset = 'trainset.h5'
hdf_validset = 'validset.h5'
hdf_testset_ol = 'testset_ol.h5'

# 从单个原始特征中提取出的特征
hdf_user_pref_cat = 'f_user_pref_cat.h5'
hdf_app_popularity = 'f_app_popularity.h5'
hdf_user_activity = 'f_user_activity.h5'
hdf_user_activity_cat = 'f_user_activity_cat.h5'
hdf_residence_cat = 'f_residence_cat.h5'
hdf_hour_weight = 'f_hour_weight.h5'

# 分析结果
hdf_pos_score_series = 'pos_score_series.h5'

# 特征名字 fn: feature name
fn_is_not_wifi = 'is_not_wifi'
fn_user_activity = 'user_activity'
fn_user_activity_cat = 'user_activity_cat'
fn_is_installed = 'is_installed'
fn_app_popularity = 'app_popularity'
fn_cat_pref = 'cat_pref'
fn_is_pref_cat = 'is_pref_cat'
fn_is_child_old = 'is_child_old'
fn_userID_click_cumcount = 'userID_click_cumcount'
fn_userID_stats_conversion_cumcount = 'userID_stats_conversion_cumcount'

# 注意用 set 而不是 list，以避免在程序中错误地重复添加
# 那些取值个数较多的特征, 依次为[677, 3447, 6315, 7219, 2595627]
dense_feature_name_set = {
    'camgaignID',
    'adID',
    'creativeID',
    'positionID',
    'userID'
}
# 不应该自动添加，手动添加更加灵活
numeric_features_set = {
    'conversion_ratio_creativeID',
    'conversion_ratio_positionID',
    'conversion_ratio_residence',
    # 'conversion_ratio_hometown',
    'conversion_ratio_positionType',
    'conversion_ratio_appCategory',
    'conversion_ratio_appID',
    'confidence',
    'conversion_ratio_age',
    'conversion_ratio_appPlatform',
    'conversion_ratio_camgaignID',
    'conversion_ratio_advertiserID',
    'conversion_ratio_adID',
    'hometown_hour_install_prob'
    'click_count_age',
    'click_count_square_age',
    'click_count_sitesetID',
    'click_count_square_sitesetID'
    # 'conversion_ratio_userID_stats'
}
# 那些无法提取 count_ratio 的 columns
columns_set_without_count_ratio = {
    'label',
    'clickTime',
    'conversionTime',
    'instanceID'
}
# trainset 与 testset_ol 取值不匹配的列, 同时这些列本身也是应当舍弃的
# 这里犯错误了，'advertiserID'是一个重要的特征
columns_set_mismatch = {
    # 'advertiserID',
    # 'camgaignID',
    # 'adID',
    # 'creativeID',
    # 'positionID',
    'userID'
}
# 不能把'label' 放到这里
columns_set_useless = {
    'clickTime',
    'conversionTime',
    'instanceID'
}
# 不能明显有助于分类的列
columns_set_inapparent = {
    'marriageStatus',
    'appPlatform',
    # 'hour',
    # 'week',
    # 'user_activity',
    'age',
    'gender',
    'education',
    'haveBaby',
    'telecomsOperator',
    'sitesetID',
    'connectionType',
    fn_is_not_wifi,
    'residence',
    'hometown',
    # 'creativeID',
    'userID_stats',
    'userID_stats_less',
    # 'appID',
    'connectionType',
    'positionID',
    'advertiserID',
    'appCategory'
    # 'camgaignID'
    # 'creativeID'
    # 'appID'
}
# 那些可以构造 click_count 特征的列
columns_set_to_construct_click_count = {
    # 'age',
    # 'userID'
    # 'sitesetID'
}
# 那些可以构造 conversion_ratio 特征的列
columns_set_to_construct_conversion_ratio = {
    # 'label',
    # 'clickTime',
    # 'conversionTime',
    # 'creativeID',
    # 'userID',
    'positionID',
    # 'connectionType',
    # 'telecomsOperator',
    # 'sitesetID',
    # 'positionType',
    # 'age',
    # 'gender',
    # 'education',
    # 'marriageStatus',
    # 'haveBaby',
    'hometown',
    # 'residence',
    'adID',
    # 'camgaignID',
    'advertiserID'
    # 'appID',
    # 'appPlatform',
    # 'appCategory'
    # 'hour',
    # 'week'
    # 'userID_stats'
}
# 那些可以构造 conversion_count 特征的列
columns_set_to_construct_conversion_count = set({
    # 'label',
    # 'clickTime',
    # 'conversionTime',
    # 'creativeID',
    # 'userID',
    # 'positionID',
    # 'connectionType',
    # 'telecomsOperator',
    # 'sitesetID',
    # 'positionType',
    # 'age',
    # 'gender',
    # 'education',
    # 'marriageStatus',
    # 'haveBaby',
    # 'hometown',
    # 'residence'
    # 'adID'
    # 'camgaignID'
    # 'advertiserID'
    # 'appID',
    # 'appPlatform'
    # 'appCategory'
    # 'hour',
    # 'week'
})
# 那些可以构造 conversion_count 特征的二次组合特征
columns_set_to_construct_conversion_count_combi = {
    # 'hometown_advertiserID',
    # 'hometown_appID',
    # 'positionID_marriageStatus'
    # 'creativeID_residence'
    # 'creativeID_age'
    # 'creativeID_connectionType'
    # 'positionID_connectionType'
}
# 那些可以构造 click_count 特征的二次组合特征
columns_set_to_construct_click_count_combi = [
    ['positionID', 'connectionType']
]
# 已被重新分类的列
columns_set_reclassified = {
    # 'residence',
    'age'
}
# 组合特征集合
combi_feature = [
    # ['appID', fn_is_not_wifi],
    ['creativeID', fn_is_not_wifi],
    # ['creativeID', 'age'],
    # ['appCategory', 'positionType'],  # 效果不好
    # ['appCategory', 'hometown'],
    # ['appCategory', 'residence'],     # 效果不好
    # ['appCategory', 'telecomsOperator'],# 无明显效果
    # ['appCategory', 'appPlatform'], # 无明显效果
    ['appCategory', 'week'],
    ['appCategory', 'connectionType'],
    # ['appCategory', 'sitesetID'],  # 无提升
    ['appCategory', 'gender'],
    # ['appCategory', 'haveBaby'],
    # ['appCategory', 'hour'], # 无明显效果
    # ['appCategory', 'age'],
    # ['appCategory', 'education'],
    # ['appCategory', 'marriageStatus']
    # ['appID', 'hour'] # 无明显效果
    # ['age', 'hour'] # 无明显效果
    # ['gender', 'age'],
    # ['week', 'hour']
    # ['appID', 'adID']
    # ['appID', 'camgaignID'] # 【误差上升5个十万分点，故不保留该特征】
    # ['telecomsOperator', 'connectionType']
    # ['creativeID', 'appID']
    ['positionID', 'connectionType'],
    ['positionID', 'advertiserID'],
    ['positionID', 'gender'],
    ['positionID', 'telecomsOperator'],
    ['positionID', 'marriageStatus'],
    ['gender', 'education'],
    # ['marriageStatus', 'residence'],
    # ['camgaignID', 'connectionType'],
    # ['age', 'telecomsOperator'],
    # ['age', 'education'],
    # ['positionID', 'residence'],
    # ['positionID', 'education'],
    # ['positionID', 'camgaignID'],
    # ['positionID', 'creativeID'],
    # ['positionID', 'hometown'],
    # ['positionID', 'age'],
    # ['positionID', 'appCategory']
    # ['hometown', 'residence']
    ['age', 'marriageStatus']
]

columns_set_to_construct_hour_prob = {
    # 'age',
    # 'gender'
    # 'education',
    # 'marriageStatus',
    # 'haveBaby',
    # 'hometown',
    # 'residence'
}

# 特征群文件
hdf_ad_fg = 'fg_ad.h5'
hdf_user_fg = 'fg_user.h5'

# 合并后的特征群文件
hdf_trainset_fg = 'fg_trainset.h5'
hdf_validset_fg = 'fg_validset.h5'
hdf_testset_ol_fg = 'fg_testset_ol.h5'

# 稀疏矩阵, 一次项
npz_ad = 'ad_csc.npz'
npz_ad_test_ol = 'ad_csc_test_ol.npz'
npz_context = 'context_csc.npz'
npz_context_test_ol = 'context_csc_test_ol.npz'
npz_user = 'user_csc.npz'
npz_user_test_ol = 'user_csc_test_ol.npz'
# 稀疏矩阵, 二次项
npz_user_ad = 'user_ad.npz'
npz_user_ad_test_ol = 'user_ad_test_ol.npz'
npz_user_context = 'user_context.npz'
npz_user_context_test_ol = 'user_context_test_ol.npz'
npz_ad_context = 'ad_context.npz'
npz_ad_context_test_ol = 'ad_context_test_ol.npz'

# ndarray
npy_y = 'y.npy'
npy_y_train = 'y_train.npy'
npy_y_valid = 'y_valid.npy'
npy_y_test = 'y_test.npy'

# 稀疏矩阵，合并后的数据
npz_X_linear = 'X_linear.npz'
npz_X_interactive = 'X_interactive.npz'
npz_X = 'X.npz'
npz_X_test_ol_linear = 'X_test_ol_linear.npz'
npz_X_test_ol_interactive = 'X_test_ol_interactive.npz'
npz_X_test_ol = 'X_test_ol.npz'

# 稀疏矩阵，建模数据
npz_X_train = 'X_train.npz'
npz_X_valid = 'X_valid.npz'
npz_X_test = 'X_test.npz'

# cv_result
csv_cv_res_xgb = 'cv_res_xgb.csv'

# 提交集
csv_submission = 'submission.csv'
csv_submission_lr = 'submission_lr.csv'
csv_submission_xgb = 'submission_xgb.csv'
csv_submission_avg = 'submission_avg.csv'
