# 路径
path_pre = '../'
path_original_dataset = path_pre + 'original-dataset/'
path_intermediate_dataset = path_pre + 'intermediate-dataset/'
path_feature = path_pre + 'feature/'
path_modeling_dataset = path_pre + 'modeling-dataset/'
path_model = path_pre + 'model/'
path_submission_dataset = path_pre + 'submission-dataset/'
path_cv_res = path_pre + 'cv-res/'

# 是否舍弃后 5 天的负样本
discard_negative_last_5_day = False

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
hdf_user_app_cat = 'user_app_cat.h5'
hdf_userID_appID_pair_installed = 'userID_appID_pair_installed.h5'
hdf_trainset = 'trainset.h5'
hdf_testset_ol = 'testset_ol.h5'

# 从单个原始特征中提取出的特征
hdf_user_pref_cat = 'f_user_pref_cat.h5'
hdf_user_cat_weight = 'f_user_cat_weight.h5'
hdf_app_popularity = 'f_app_popularity.h5'
hdf_user_activity = 'f_user_activity.h5'
hdf_user_activity_cat = 'f_user_activity_cat.h5'

hdf_hour_weight = 'f_hour_weight.h5'
hdf_conversion_ratio_connectionType = 'f_conversion_ratio_connectionType.h5'
hdf_conversion_ratio_telecomsOperator = 'f_conversion_ratio_telecomsOperator.h5'
hdf_hour = 'f_hour.h5'
hdf_week = 'f_week.h5'

hdf_userID = 'f_userID.h5'

# 特征名字 fn: feature name
fn_is_not_wifi = 'is_not_wifi'
fn_user_activity = 'user_activity'
fn_user_activity_cat = 'user_activity_cat'
fn_is_installed = 'is_installed'
fn_app_popularity = 'app_popularity'

fn_age_connectionType = 'age_connectionType'
fn_gender_connectionType = 'gender_connectionType'
fn_education_connectionType = 'education_connectionType'
fn_marriageStatus_connectionType = 'marriageStatus_connectionType'
fn_residence_connectionType = 'residence_connectionType'
fn_appCategory_connectionType = 'appCategory_connectionType'
fn_connectionType_telecomsOperator = 'connectionType_telecomsOperator'
fn_appID_connectionType = 'appID_connectionType'
fn_appID_is_wifi = 'appID_is_wifi'

fn_age_appCategory = 'age_appCategory'
fn_gender_appCategory = 'gender_appCategory'
fn_education_appCategory = 'education_appCategory'
fn_marriageStatus_appCategory = 'marriageStatus_appCategory'
fn_haveBaby_appCategory = 'haveBaby_appCategory'

fn_cat_pref = 'cat_pref'
fn_is_pref_cat = 'is_pref_cat'
fn_is_child_old = 'is_child_old'

# 注意用 set 而不是 list，以避免在程序中错误地重复添加
# 那些取值个数较多的特征, 依次为[677, 3447, 6315, 7219, 2595627]
dense_feature_name_set = {
    'camgaignID',
    'adID',
    'creativeID',
    'positionID',
    'userID'
}
boolean_features_set = {
    fn_is_installed,
    fn_is_pref_cat,
    fn_is_not_wifi,
    fn_is_child_old
}
# 不应该手动添加，而应该在构造该特征时自动添加
numeric_features_static_set = {
    'confidence'
}
hdf_numeric_features_set = 'numeric_features.h5'
# 那些无法提取 count_ratio 的 columns
columns_set_without_count_ratio = {
    'label',
    'clickTime',
    'conversionTime',
    'instanceID'
}
# trainset 与 testset_ol 取值不匹配的列, 同时这些列本身也是应当舍弃的
columns_set_mismatch = {
    'advertiserID',
    'camgaignID',
    'adID',
    'creativeID',
    'positionID',
    'userID'
}
# 不构造转化率的列
columns_set_without_conversion_ratio = {
    'positionType',
    'haveBaby',
    'education'
}
columns_set_useless = {
    'label',
    'clickTime',
    'conversionTime',
    'instanceID'
}
# 不能明显有助于分类的列
columns_set_inapparent = {
    'marriageStatus',
    'appPlatform',
    'hour',
    'week',
    'conversion_ratio_marriageStatus',
    'conversion_ratio_hour',
    'conversion_ratio_week',
    'user_activity',
    'age'
}

# 特征群文件
hdf_context_dataset_fg = 'fg_context_dataset.h5'
hdf_context_testset_ol_fg = 'fg_context_testset_ol.h5'
hdf_ad_fg = 'fg_ad.h5'
hdf_user_fg = 'fg_user.h5'

# 合并后的特征群文件
hdf_trainset_fg = 'fg_trainset.h5'
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
npz_X_test = 'X_test.npz'

# cv_result
csv_cv_res_xgb = 'cv_res_xgb.csv'

# 提交集
csv_submission = 'submission.csv'
csv_submission_lr = 'submission_lr.csv'
csv_submission_xgb = 'submission_xgb.csv'
csv_submission_avg = 'submission_avg.csv'

