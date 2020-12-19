import sys
sys.path.append("..")
import laimbionet.train as train
import os
from laimbionet.networks import unet,DeepLab, no_pool_3d, unet_3dpatch,deeplab_3d, deep_lab_small_3d,U_net3D_residuals_atrous
from laimbionet.input import Dataset, Experiment
import tensorflow as tf

#training script



#path to subjects images
path_subjects = 'paris'


#keyword of the images used in the network, just peaces of the names are needed
key_words_in = ["_MRI"]
key_words_out = ["_CT"]
key_words_roi = ['_Mask']



my_dataset = Dataset(path_subjects, key_words_in, key_words_out, key_words_roi)

print(my_dataset.get_groups_keys())

path_experiment = os.path.join( "experiment")

my_experiment = Experiment(path_experiment)

#set a session to train this create all the folders and variables needed
# the flags continue_session and clean_up are needed to avoid miss deleting a whole network session without intention

network_name = 'unet3d_residuals_64'
# network_name = 'no_pool_32'
# network_name = 'simple_deeplab_32_prueba_borrar'

my_experiment.set_session(network_name,
                          session_data=my_dataset.get_data_from_groups(['train','test'] )
                          , train_name='train', test_name='test', continue_session=False, clean_up=True)



print(my_experiment)

#just a cuda flag to force the use of a certain gpu, it shouldn't be needed if you have just one gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#object to train the network
my_trainer = train.RegressionTrainer_patch3d()


# network = unet_3dpatch.Unet
network = U_net3D_residuals_atrous.Unet3d_residuals
# network = deeplab_3d.Network_DeepLab_v3_xception
# network = no_pool_3d.No_pool_net_3d
# network = deep_lab_small_3d.Network_DeepLab_v3_xception
epochs = 25
#
# args_network = {'name': network_name , 'input_channels': 1, 'filters': 32, 'output_stride': 4,
# 'middle_blocks': 8, 'aspp_blocks': 1, 'renorm' : True}

args_network = {'name': network_name , 'input_channels': 1, 'filters': 64}



my_trainer.network = network

my_trainer.args_network = args_network
my_trainer.use_roi = True
my_trainer.error_function = tf.losses.absolute_difference
my_trainer.batch_size = 2
my_trainer.shuffle_buffer = 100
my_trainer.learning_rate = 1e-4
# val size in batches
my_trainer.val_size = None
my_trainer.test_as_val = True
my_trainer.test = True
my_trainer.epochs = epochs
my_trainer.set_experiment(my_experiment)
my_trainer.write_images_tb = False


my_trainer.run()
