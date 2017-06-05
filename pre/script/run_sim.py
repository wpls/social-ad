import prepare as pp
import feature_construction as fc
import modeling as md

# 计时开始
from time import time

start = time()

pp.prepare_dataset_all()
fc.construct_feature()
# md.one_hot()

md.tuning_hyper_parameters_lr_sim_tmp(fast=True)
# md.predict_test_ol_lr()

print('Running complete.')
print('\nThe total time : {0:.0f} s'.format(time() - start))
