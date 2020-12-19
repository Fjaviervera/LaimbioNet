import sys

sys.path.append("..")
import laimbionet.train as train
import os
from laimbionet.networks import unet, DeepLab, no_pool_net, unet_3dpatch, no_pool_3d, U_net3D_residuals_atrous
from laimbionet.input import Dataset, Experiment, TfrecordCreator, ProductionCreator, ProductionCreator_patch3d, NN_OUT, \
    NN_ROI
import tensorflow as tf
from laimbionet.production import ProductionNetwork, ProductionNetwork_patch3d
import numpy as np

#path to subjects images

path_subjects = 'paris'


group = 'test'

#keyword of the images used in the network, just peaces of the names are needed
key_words_in = ["_MRI"]
key_words_out = ["_CT"]
key_words_roi = ['_Mask']
# key_words_in = ["MRI_rec"]
# key_words_out = ["CT_rec"]
# key_words_roi = ['Mask_rec']

my_dataset = Dataset(path_subjects, key_words_in, key_words_out, key_words_roi)

print(my_dataset.get_groups_keys())

path_experiment = os.path.join("experiment")

my_experiment = Experiment(path_experiment)

# set a session to train this create all the folders and variables needed
# the flags continue_session and clean_up are needed to avoid miss deleting a whole network session without intention


network_name = 'unet3d_residuals_64'

# network_name = 'no_pool_32'

# network = unet_3dpatch.Unet
network = U_net3D_residuals_atrous.Unet3d_residuals
# network = no_pool_3d.No_pool_net_3d
#

my_experiment.set_session(network_name,
                          session_data=my_dataset.get_data_from_groups(['train','test'])
                          , train_name='train', test_name='test', continue_session=True, clean_up=False)

print(my_experiment)

# just a cuda flag to force the use of a certain gpu, it shouldn't be needed if you have just one gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#
# args_network = {'name': network_name , 'input_channels': 1, 'filters': 32, 'output_stride': 8,
# 'middle_blocks': 16, 'aspp_blocks': 1}

args_network = {'name': network_name, 'input_channels': 1, 'filters': 64}


# generate slices from the the test subject in the fold
my_patch_creator = ProductionCreator_patch3d(my_dataset.get_data_from_groups())

my_patch_creator.dst_path = my_experiment.get_production_session_path()
my_patch_creator.set_dtypes(dtype_out='float')

my_patch_creator.network_side_label_check = NN_ROI
my_patch_creator.z_correction = False
my_patch_creator.standar_maxmin = False
my_patch_creator.stride_cube =16
my_patch_creator.check_valid_mask = True
my_patch_creator.patch_shape = [32,32,32]
my_patch_creator.valid_in_mask = (
(int(my_patch_creator.patch_shape[0] / 2 - my_patch_creator.stride_cube / 2), int(my_patch_creator.patch_shape[0] / 2 + my_patch_creator.stride_cube / 2)),
(int(my_patch_creator.patch_shape[1] / 2 - my_patch_creator.stride_cube / 2), int(my_patch_creator.patch_shape[1] / 2 + my_patch_creator.stride_cube / 2)),
(int(my_patch_creator.patch_shape[2] / 2 - my_patch_creator.stride_cube / 2), int(my_patch_creator.patch_shape[2] / 2 + my_patch_creator.stride_cube / 2)))


# run network in production mode
my_production = ProductionNetwork_patch3d()
my_production.network = network
my_production.model_epoch = 20
my_production.args_network = args_network


my_production.use_roi = True
my_production.initialize(my_experiment)

# error_list = []

# generate the nifti with the output of every subject in the test of the cross validation
subjects = my_dataset.get_subjects_keys(group)


for subject in subjects:

    vol_created = False




    for numpy_in, numpy_out, numpy_roi, index, shape in my_patch_creator.get_next_slice(group, subject):

        if vol_created == False:
            vol_out = np.zeros(shape, dtype=np.float32)
            vol_created = True

        numpy_out_net = my_production.produce_output_from_patch3d(numpy_in, numpy_roi)

        vol_out = my_patch_creator.fill_custom_cube(numpy_out_net, index, vol_out,custom_shape=16)

        print(index)

    vol_out = vol_out[my_patch_creator.patch_shape[0]:-my_patch_creator.patch_shape[0],
              my_patch_creator.patch_shape[1]:-my_patch_creator.patch_shape[1],
              my_patch_creator.patch_shape[2]:-my_patch_creator.patch_shape[2]]


    vol_out = my_patch_creator.correct_background(vol_out, group, subject)
    my_patch_creator.create_nifti_from_cube(vol_out, subject,group = group,meta_datos = False)

    print('%s done' % subject)
