import sys
sys.path.append("..")
from laimbionet.input import Dataset, Experiment, TfrecordCreator_patch3d, NN_OUT, NN_ROI
import os


#path to subjects images
root_path = 'paris'
path_subjects = os.path.join(root_path)


#keyword of the images used in the network, just peaces of the names are needed
key_words_in = ["_MRI"]
key_words_out = ["_CT"]
key_words_roi = ['_Mask']

#creation of the dataset containing the images
my_dataset = Dataset(path_subjects, key_words_in, key_words_out, key_words_roi)

print(my_dataset.get_groups_keys())


#Creation of the experiment in which the tfrecords are stored
path_experiment =  "experiment"
my_experiment = Experiment(path_experiment)


#object to create the tfrecord
#get_data_from_groups import the data that we want to use in the tfrecord
my_tf_creator = TfrecordCreator_patch3d(my_dataset.get_data_from_groups(['train','test'] ))

#we can transform the data into mean 0 standard deviation 1
# my_tf_creator.calc_z_correction('train',use_roi= True)


#data augmentation rotate and move the slices
my_tf_creator.data_augmentation = True
my_tf_creator.standar_maxmin = False

# my_tf_creator.set_data_augmentation(angle = 360)
#set what kind of data augmentation is used
#my_tf_creator.set_data_augmentation(dx=5,dy=5,angle=30)

#if we want to shuffle the slices in the tfrecord
my_tf_creator.shuffle = True
#destination of the tfrecords
my_tf_creator.dst_path = my_experiment.get_records_path()
#type of output of the network in this case regression so float
my_tf_creator.set_dtypes(dtype_out='float')
#we can check if in the output we always want something, there must be cases in which the output is all zero
# for example that could happen if the mri as input and the ct as output have different length
my_tf_creator.network_side_label_check = NN_ROI
# finally create the tfrecord, subject buffer depend on how must ram memory do you have available.
# the shuffle will be done in those subjects.
my_tf_creator.padding = 32
my_tf_creator.group_stride_dict= { 'train':16,'test':16}
my_tf_creator.run(subjects_buffer_size=16)

