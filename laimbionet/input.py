from scipy.ndimage import interpolation
import os
import random
import tensorflow as tf
import numpy as np
import nibabel as nib
import copy
import pprint
import logging
from random import shuffle
import glob
import gc

try:
    import medpy.io

    medpy_found = True
except ImportError:
    medpy_found = False

    pass

try:
    import cv2

    cv2_found = True
except ImportError:
    cv2_found = False

    pass

NN_IN = 'network_in'
NN_OUT = 'network_out'
NN_ROI = 'network_roi'

CV_TRAIN = 'train_cv'
CV_TEST = 'test_cv'


def list_nohidden_directories(path):
    if os.name == 'nt':
        import win32api, win32con

    def file_is_hidden(p):
        if os.name == 'nt':
            attribute = win32api.GetFileAttributes(p)
            return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
        else:
            return p.startswith('.')  # linux-osx

    file_list = [f for f in sorted(os.listdir(path)) if (not file_is_hidden(os.path.join(path, f)) and os.path.isdir(os.path.join(path, f)))]
    return file_list


def list_nohidden_files(path):
    if os.name == 'nt':
        import win32api, win32con

    def file_is_hidden(p):
        if os.name == 'nt':
            attribute = win32api.GetFileAttributes(p)
            return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
        else:
            return p.startswith('.')  # linux-osx

    file_list = [f for f in glob.glob(path + "/**/*", recursive=True) if
                 (not file_is_hidden(f) and not os.path.isdir(os.path.join(path, f)))]
    return file_list


class Data_class():
    def __init__(self, group_subject_list, output_dtype='float'):

        self.group_subject_list = group_subject_list
        self.output_dtype = output_dtype

    def reading_fun(self, subject_index):

        tensor_flair, tensor_lesions = tf.py_function(func=self.generate_cube, inp=[subject_index],
                                                      Tout=[tf.float32, tf.float32])

        return tensor_flair, tensor_lesions

    def generate_cube(self, subject_index):

        dict_subject = self.group_subject_list[subject_index]

        index = self.random_index_from_list(dict_subject['index_list'])

        cube_input = self.get_cube_from_index(dict_subject['subject_data_dict'][NN_IN], index)

        cube_output = self.get_cube_from_index(dict_subject['subject_data_dict'][NN_OUT], index)

        cube_input, cube_output = self.data_augmentation(cube_input, cube_output, self.output_dtype)

        return cube_input, cube_output

    @staticmethod
    def random_index_from_list(index_list):

        random_index = random.randint(0, len(index_list[0]) - 1)

        return (index_list[0][random_index], index_list[1][random_index], index_list[2][random_index])

    @staticmethod
    def get_cube_from_index(data, index, cube_shape=(32, 32, 32)):

        return data[int(index[0] - cube_shape[0] / 2):int(index[0] + cube_shape[0] / 2),
               int(index[1] - cube_shape[1] / 2):int(index[1] + cube_shape[1] / 2),
               int(index[2] - cube_shape[2] / 2):int(index[2] + cube_shape[2] / 2), :]

    @staticmethod
    def data_augmentation(input, output, output_dtype):

        angle = random.randint(0, 180)

        input = interpolation.rotate(input, angle, axes=(1, 0), reshape=False, mode='nearest')

        if output_dtype == 'int':
            output = interpolation.rotate(output, angle, reshape=False, mode='nearest', order=0)
        else:
            output = interpolation.rotate(output, angle, reshape=False, mode='nearest')

        return input, output


class DataOperator:
    def __init__(self, dictionary):

        self._dictionary = dictionary

    def __str__(self):
        return pprint.pformat(self._dictionary)

    def __copy__(self):
        return DataOperator(copy.deepcopy(self._dictionary))

    def get_network_sides(self, group):
        return list(self._dictionary[group].keys())

    def get_list_subjects(self, group):
        # we should have the same subject in every channel so just read channel 1
        return list(self._dictionary[group][NN_IN][1].keys())

    def get_list_channels(self, group, network_side):
        return list(self._dictionary[group][network_side].keys())

    def get_list_groups(self):
        return list(self._dictionary.keys())

    def get_subject_path(self, group, network_side, channel, subject):
        return self._dictionary[group][network_side][channel][subject]

    def get_groups_dictionary(self, groups):

        if isinstance(groups, str):
            groups_list = [groups]
        elif isinstance(groups, list):
            groups_list = groups
        else:
            raise ValueError(' Error: "groups" must be a string or a list.')

        new_dict = dict()
        for group in groups_list:
            new_dict[group] = dict(self._dictionary[group])
        return copy.deepcopy(new_dict)

    def get_dictionary(self):
        return copy.deepcopy(self._dictionary)

    def get_data(self, groups='all'):

        if not isinstance(groups, str) and not isinstance(groups, list) and not isinstance(groups, tuple):
            raise ValueError(' get_data must recieve a string or a list')

        if groups is 'all':
            return self.__copy__()
        else:
            return DataOperator(self.get_groups_dictionary(groups))


class TfrecordCreator:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self.dst_path = ''
        self._shapes_tfrecord = dict()
        self.z_correction = False
        self.data_augmentation = False
        self.slice_dim = 2
        self.means_z = None
        self.stds_z = None
        self._used_roi_z_correction = False
        self.shuffle = False
        self.standar_maxmin = False
        self.network_side_label_check = None
        self.slices_ba = 0

        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._DX = 5
        self._DY = 5
        self._ANGLE = 30

        self._read_function = None
        self._Z_CORR = False

    def _print_info(self, group):

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(os.path.join(self.dst_path, group + '.log'), 'w'))
        print = logger.info
        print('#######TFRECORD CREATION INFO#######################')
        print('-Dictionary of data: ')
        print(self.dictionary)
        print('-Group: %s' % group)
        print('-Shape tfrecord: %s' % self._shapes_tfrecord)
        print('-Z correction: %r' % self.z_correction)
        print('-Data augmentation: %r' % self.data_augmentation)
        print('-Slice dimension: %d' % self.slice_dim)
        print('-Means for Z correction: %s' % self.means_z)
        print('-Stds for Z correction: %s' % self.stds_z)
        print('-Use ROI in z correction: %r' % self._used_roi_z_correction)
        print('-Shuffle: %r' % self.shuffle)
        print('-NN_IN_DTYPE: %s' % self._NN_IN_DTYPE)
        print('-NN_OUT_DTYPE: %s' % self._NN_OUT_DTYPE)
        print('-NN_ROI_DTYPE: %s' % self._NN_ROI_DTYPE)
        print('DX for data aug: %d' % self._DX)
        print('DY for data aug: %d' % self._DY)
        print('ANGLE for data aug: %d' % self._ANGLE)
        print('Read function used: %s' % self._read_function)
        print('###################################################')
        logger.handlers.pop()

    @staticmethod
    def _check_valid_output_slice(network_vols3d_subject, slice_id, network_side):

        slice_mask = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
            np.float32)

        if np.sum(slice_mask) > 0:
            contains_labels = True
        else:
            contains_labels = False
        return contains_labels

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol =  np.squeeze(img.get_data())

        return vol

    def resize_slices(self, new_size, group='all_groups', network_side=None):
        if group is 'all_groups':
            groups_to_do = self.dictionary.get_list_groups()
        else:
            groups_to_do = [group]

        if network_side is None:
            for group in groups_to_do:
                for network_side in self.dictionary.get_network_sides(group):
                    self._set_size_side(group, network_side, new_size)
        else:
            for group in groups_to_do:
                self._set_size_side(group, network_side, new_size)

    def _set_size_side(self, group, network_side, new_size):

        if not isinstance(new_size, tuple):
            raise ValueError('Error: "new_shape" must be a tuple')

        if len(new_size) != 2:
            raise ValueError('Error: "new_shape" must have two values')

        if network_side not in self.dictionary.get_network_sides(group):
            raise ValueError('Error: %s is not a network side')

        self._shapes_tfrecord[network_side] = new_size

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)
        # Always perform a np.rollaxis, we want the slicing position last
        if self.slice_dim != 2:
            vol = np.rollaxis(vol, self.slice_dim, 3)

        if self.standar_maxmin:
            vol = (vol - vol.min()) / (vol.max() - vol.min())

        if self.slices_ba != 0:
            vol = np.pad(vol, ((0, 0), (0, 0), (self.slices_ba, self.slices_ba)), 'edge')

        return vol

    @staticmethod
    def _resize_slice(slice_image, newsize, inter):
        if cv2_found:
            if inter is 'float':
                inter = cv2.INTER_CUBIC
            elif inter is 'int':
                inter = cv2.INTER_NEAREST

            slice_image = cv2.resize(slice_image, newsize,
                                     interpolation=inter)
            if slice_image.ndim is 2:
                slice_image = np.expand_dims(slice_image, axis=-1)

        else:
            raise ValueError(
                ' CV2 is not installed and is needed for resize slices, to install it use "sudo pip install opencv-python"')
        return slice_image

    def calc_z_correction(self, group, use_roi=False):

        # apriori the z_corr is only for network_in

        # INPUTS:
        # group: string that contains the group name that is going to be used to calculate the values for the z correction
        # use_roi: Boolean used to whether use a ROI to calculate the correction or not. If ROI channels != In channels
        # just the first roi channel is used
        # OUTPUT:Means and stds for z correction in a list in channel order

        means_per_channel = []
        stds_per_channel = []
        channel_list = self.dictionary.get_list_channels(group, NN_IN)
        subject_list = self.dictionary.get_list_subjects(group)
        for channel in channel_list:
            vol_list_flatten = []
            print('channel %d' % channel)
            for subject in subject_list:

                vol_subject = self._read_data(self.dictionary.get_subject_path(group, NN_IN, channel, subject))
                # print(subject)
                # print(np.max(vol_subject))
                # print(np.min(vol_subject))

                if use_roi:
                    self._used_roi_z_correction = True
                    if len(self.dictionary.get_list_channels(group, NN_IN)) == len(
                            self.dictionary.get_list_channels(group, NN_ROI)):

                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, channel, subject))

                    else:
                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, 1, subject))

                    vol_list_flatten.append(np.extract(roi_subject, vol_subject))
                    print(np.mean(np.extract(roi_subject, vol_subject)))
                else:

                    vol_list_flatten.append(vol_subject.flatten())

            data_for_scale = np.concatenate(vol_list_flatten)

            means_per_channel.append(np.mean(data_for_scale))
            stds_per_channel.append(np.std(data_for_scale))

        self.means_z = means_per_channel
        self.stds_z = stds_per_channel

        self._Z_CORR = True
        print(means_per_channel)
        print(stds_per_channel)
        return means_per_channel, stds_per_channel

    def set_z_correction(self, means_z, stds_z):

        self.means_z = means_z

        self.stds_z = stds_z

        self._Z_CORR = True

    def _create_tfrecord_writer(self, group):
        return tf.python_io.TFRecordWriter(os.path.join(self.dst_path, group + '.tfrecord'))

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))

                    vol = np.expand_dims(vol, axis=-1)
                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=vol_channels_list[0].ndim - 1)
        return network_vols3d_subject

    def _convert_to_serial_and_write(self, group, data_list, tfrecord_writer):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        num_examples = len(data_list)

        sample_indices = list(range(num_examples))
        if self.shuffle:
            print('Doing shuffle!')
            random.shuffle(sample_indices)

        side_words = ['in', 'out', 'roi']

        for index in sample_indices:
            sample_features = dict()
            for network_side, shape_side_word in zip(self.dictionary.get_network_sides(group), side_words):
                sample_features[network_side] = _bytes_feature(data_list[index][network_side].tostring())
                rows, cols, channels = data_list[index][network_side].shape
                sample_features['rows_' + shape_side_word] = _int64_feature(rows)
                sample_features['cols_' + shape_side_word] = _int64_feature(cols)
                sample_features['channels_' + shape_side_word] = _int64_feature(channels)

            example = tf.train.Example(features=tf.train.Features(feature=sample_features))
            tfrecord_writer.write(example.SerializeToString())





    def set_data_augmentation(self, dx=5, dy=5, angle=360):

        self._DX = dx
        self._DY = dy
        self._ANGLE = angle

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    def _do_data_augmentation(self, group, slice_data_dict):

        dx = random.randrange(0, self._DX)
        dy = random.randrange(0, self._DY)
        angle = random.randrange(0, self._ANGLE)

        for network_side in self.dictionary.get_network_sides(group):

            dtype = self.dtypes_dict[network_side]

            if slice_data_dict[network_side].ndim > 3:
                slice_data_dict[network_side] = np.squeeze(slice_data_dict[network_side])
            if slice_data_dict[network_side].ndim < 3:
                slice_data_dict[network_side] = np.expand_dims(slice_data_dict[network_side], axis=-1)

            if dtype is 'float':

                slice_data_dict[network_side] = interpolation.shift(slice_data_dict[network_side], [dx, dy, 0],
                                                                    mode='nearest')
                slice_data_dict[network_side] = interpolation.rotate(slice_data_dict[network_side], angle,
                                                                     reshape=False,
                                                                     mode='nearest')
            elif dtype is 'int':
                slice_data_dict[network_side] = interpolation.shift(slice_data_dict[network_side], [dx, dy, 0],
                                                                    mode='nearest', order=0)
                slice_data_dict[network_side] = interpolation.rotate(slice_data_dict[network_side], angle,
                                                                     reshape=False,
                                                                     mode='nearest', order=0)
            else:
                raise ValueError('Error: "dtype" in %s not recognized, %s' % (network_side, dtype))

        return slice_data_dict

    def _list_slices_subject(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        _, _, slices, _ = network_vols3d_subject[NN_IN].shape

        tfrecord_slice_list = []
        # HE PUESTO UN OFFSET PARA LA PELVIS EN RESO
        for slice_id in range(self.slices_ba, slices - self.slices_ba - 20):

            if self.network_side_label_check:
                if not self._check_valid_output_slice(network_vols3d_subject, slice_id, self.network_side_label_check):
                    continue

            slice_data_dict = dict()

            for network_side in self.dictionary.get_network_sides(group):

                if network_side in list(self._shapes_tfrecord.keys()) and \
                        network_vols3d_subject[network_side][:, :,
                        (slice_id - self.slices_ba):(slice_id + self.slices_ba + 1), :].shape[0:2] != \
                        self._shapes_tfrecord[
                            network_side]:

                    slice_data_dict[network_side] = self._resize_slice(
                        network_vols3d_subject[network_side][:, :,
                        (slice_id - self.slices_ba):(slice_id + self.slices_ba + 1), :]
                        , self._shapes_tfrecord[network_side], self.dtypes_dict[network_side]).astype(np.float32)

                else:
                    slice_data_dict[network_side] = network_vols3d_subject[network_side][:, :,
                                                    (slice_id - self.slices_ba):(slice_id + self.slices_ba + 1),
                                                    :].astype(
                        np.float32)

            if self.z_correction:
                if self._Z_CORR:

                    slice_data_dict[NN_IN] = ((slice_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                        np.float32)
                else:
                    raise ValueError(
                        'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                        or they must be sat manually in the object')

            if self.data_augmentation:
                slice_data_dict = self._do_data_augmentation(group, slice_data_dict)

            tfrecord_slice_list.append(slice_data_dict)

        return tfrecord_slice_list

    def run(self, subjects_buffer_size=1):

        if not isinstance(subjects_buffer_size, int):
            raise ValueError(' Error: "subjects_buffer_size" must be an integer.')

        groups_to_tfrecord = self.dictionary.get_list_groups()

        for group in groups_to_tfrecord:

            print('group %s' % group)

            writer = self._create_tfrecord_writer(group)
            subjects = self.dictionary.get_list_subjects(group)
            shuffle(subjects)
            for subject_id in range(0, len(subjects), subjects_buffer_size):

                subjects_buffer = subjects[subject_id:subject_id + subjects_buffer_size]
                list_slices_buffer = []
                for subject in subjects_buffer:
                    print('subject %s' % subject)

                    list_slices_buffer = list_slices_buffer + self._list_slices_subject(group, subject)

                self._convert_to_serial_and_write(group, list_slices_buffer
                                                  , writer)
            self._print_info(group)


class TfrecordCreator_patch3d:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self.dst_path = ''
        self._shapes_tfrecord = dict()
        self.set_patch_shape((32, 32, 32), group='all_groups', network_side=None)
        self.z_correction = False
        self.data_augmentation = False
        self.means_z = None
        self.stds_z = None
        self._used_roi_z_correction = False
        self.shuffle = False
        self.standar_maxmin = False
        self.network_side_label_check = None
        # self.stride_cube = 4
        self.group_stride_dict = {}
        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._DX = 5
        self._DY = 5
        self._ANGLE = 30

        self._read_function = None
        self._Z_CORR = False

    def _print_info(self, group):

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(os.path.join(self.dst_path, group + '.log'), 'w'))
        print = logger.info
        print('#######TFRECORD CREATION INFO#######################')
        print('-Dictionary of data: ')
        print(self.dictionary)
        print('-Group: %s' % group)
        print('-Shape tfrecord: %s' % self._shapes_tfrecord)
        print('-Z correction: %r' % self.z_correction)
        print('-Data augmentation: %r' % self.data_augmentation)
        print('-Means for Z correction: %s' % self.means_z)
        print('-Stds for Z correction: %s' % self.stds_z)
        print('-Use ROI in z correction: %r' % self._used_roi_z_correction)
        print('-Shuffle: %r' % self.shuffle)
        print('-NN_IN_DTYPE: %s' % self._NN_IN_DTYPE)
        print('-NN_OUT_DTYPE: %s' % self._NN_OUT_DTYPE)
        print('-NN_ROI_DTYPE: %s' % self._NN_ROI_DTYPE)
        print('DX for data aug: %d' % self._DX)
        print('DY for data aug: %d' % self._DY)
        print('ANGLE for data aug: %d' % self._ANGLE)
        print('Read function used: %s' % self._read_function)
        print('###################################################')
        logger.handlers.pop()

    @staticmethod
    def _check_valid_output_cube3d(network_vols3d_subject, cube_id, network_side, patch_shape):

        x_i, y_i, z_i = cube_id

        cube_mask = network_vols3d_subject[network_side][(x_i):(x_i + patch_shape[0]), (y_i):(y_i + patch_shape[1]),
                    (z_i):(z_i + patch_shape[2])].astype(
            np.float32)

        if np.sum(cube_mask) > 0:
            contains_labels = True
        else:
            contains_labels = False
        return contains_labels

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol = np.squeeze(img.get_data())

        return vol

    def set_patch_shape(self, new_size, group='all_groups', network_side=None):
        if group is 'all_groups':
            groups_to_do = self.dictionary.get_list_groups()
        else:
            groups_to_do = [group]

        if network_side is None:
            for group in groups_to_do:
                for network_side in self.dictionary.get_network_sides(group):
                    self._set_size_side(group, network_side, new_size)
        else:
            for group in groups_to_do:
                self._set_size_side(group, network_side, new_size)

    def _set_size_side(self, group, network_side, new_size):

        if not isinstance(new_size, tuple):
            raise ValueError('Error: "new_shape" must be a tuple')

        if len(new_size) != 3:
            raise ValueError('Error: "new_shape" must have three values')

        if network_side not in self.dictionary.get_network_sides(group):
            raise ValueError('Error: %s is not a network side')

        self._shapes_tfrecord[network_side] = new_size

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)

        if self.standar_maxmin:
            vol = (vol - vol.min()) / (vol.max() - vol.min())

        return vol

    @staticmethod
    def _resize_slice(slice_image, newsize, inter):
        if cv2_found:
            if inter is 'float':
                inter = cv2.INTER_CUBIC
            elif inter is 'int':
                inter = cv2.INTER_NEAREST

            slice_image = cv2.resize(slice_image, newsize,
                                     interpolation=inter)
            if slice_image.ndim is 2:
                slice_image = np.expand_dims(slice_image, axis=-1)

        else:
            raise ValueError(
                ' CV2 is not installed and is needed for resize slices, to install it use "sudo pip install opencv-python"')
        return slice_image

    def calc_z_correction(self, group, use_roi=False):

        # apriori the z_corr is only for network_in

        # INPUTS:
        # group: string that contains the group name that is going to be used to calculate the values for the z correction
        # use_roi: Boolean used to whether use a ROI to calculate the correction or not. If ROI channels != In channels
        # just the first roi channel is used
        # OUTPUT:Means and stds for z correction in a list in channel order

        means_per_channel = []
        stds_per_channel = []
        channel_list = self.dictionary.get_list_channels(group, NN_IN)
        subject_list = self.dictionary.get_list_subjects(group)
        for channel in channel_list:
            vol_list_flatten = []
            print('channel %d' % channel)
            for subject in subject_list:

                vol_subject = self._read_data(self.dictionary.get_subject_path(group, NN_IN, channel, subject))
                # print(subject)
                # print(np.max(vol_subject))
                # print(np.min(vol_subject))

                if use_roi:
                    self._used_roi_z_correction = True
                    if len(self.dictionary.get_list_channels(group, NN_IN)) == len(
                            self.dictionary.get_list_channels(group, NN_ROI)):

                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, channel, subject))

                    else:
                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, 1, subject))

                    vol_list_flatten.append(np.extract(roi_subject, vol_subject))
                    print(np.mean(np.extract(roi_subject, vol_subject)))
                else:

                    vol_list_flatten.append(vol_subject.flatten())

            data_for_scale = np.concatenate(vol_list_flatten)

            means_per_channel.append(np.mean(data_for_scale))
            stds_per_channel.append(np.std(data_for_scale))

        self.means_z = means_per_channel
        self.stds_z = stds_per_channel

        self._Z_CORR = True
        print(means_per_channel)
        print(stds_per_channel)
        return means_per_channel, stds_per_channel

    def set_z_correction(self, means_z, stds_z):

        self.means_z = means_z

        self.stds_z = stds_z

        self._Z_CORR = True

    def _create_tfrecord_writer(self, group):
        return tf.python_io.TFRecordWriter(os.path.join(self.dst_path, group + '.tfrecord'))

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))

                    vol = np.expand_dims(vol, axis=-1)

                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=vol_channels_list[0].ndim - 1)
        return network_vols3d_subject

    def _convert_to_serial_and_write(self, group, data_list, tfrecord_writer):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        num_examples = len(data_list)

        sample_indices = list(range(num_examples))
        if self.shuffle:
            print('Doing shuffle!')
            random.shuffle(sample_indices)

        side_words = ['in', 'out', 'roi']

        for index in sample_indices:
            sample_features = dict()
            for network_side, shape_side_word in zip(self.dictionary.get_network_sides(group), side_words):
                sample_features[network_side] = _bytes_feature(data_list[index][network_side].tostring())
                rows, cols, depth, channels = data_list[index][network_side].shape
                sample_features['rows_' + shape_side_word] = _int64_feature(rows)
                sample_features['cols_' + shape_side_word] = _int64_feature(cols)
                sample_features['depth_' + shape_side_word] = _int64_feature(depth)
                sample_features['channels_' + shape_side_word] = _int64_feature(channels)

            example = tf.train.Example(features=tf.train.Features(feature=sample_features))
            tfrecord_writer.write(example.SerializeToString())

        data_list = None

    def set_data_augmentation(self, dx=5, dy=5, angle=360):

        self._DX = dx
        self._DY = dy
        self._ANGLE = angle

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    def _do_data_augmentation(self, group, slice_data_dict):

        angle = random.randrange(0, self._ANGLE)

        for network_side in self.dictionary.get_network_sides(group):

            dtype = self.dtypes_dict[network_side]

            if dtype is 'float':

                slice_data_dict[network_side] = interpolation.rotate(slice_data_dict[network_side], angle,
                                                                     reshape=False,
                                                                     mode='nearest')
            elif dtype is 'int':

                slice_data_dict[network_side] = interpolation.rotate(slice_data_dict[network_side], angle,
                                                                     reshape=False,
                                                                     mode='nearest', order=0)
            else:
                raise ValueError('Error: "dtype" in %s not recognized, %s' % (network_side, dtype))

        return slice_data_dict

    def _list_cubes_subject(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        x, y, z, _ = network_vols3d_subject[NN_IN].shape

        tfrecord_cube_list = []

        for x_i in range(0, x, int(self.group_stride_dict[group])):
            for y_i in range(0, y, int(self.group_stride_dict[group])):
                for z_i in range(0, z, int(self.group_stride_dict[group])):
                    cube_id = (x_i, y_i, z_i)

                    if self.network_side_label_check:
                        if not self._check_valid_output_cube3d(network_vols3d_subject, cube_id,
                                                               self.network_side_label_check,
                                                               self._shapes_tfrecord[self.network_side_label_check]):
                            continue

                    cube_data_dict = dict()

                    for network_side in self.dictionary.get_network_sides(group):

                        x_patch, y_patch, z_patch = self._shapes_tfrecord[network_side]

                        if network_side in list(self._shapes_tfrecord.keys()) and \
                                network_vols3d_subject[network_side][(x_i):(x_i + x_patch), (y_i):(y_i + y_patch),
                                (z_i):(z_i + z_patch)].shape[0:3] != self._shapes_tfrecord[network_side]:
                            break


                        else:
                            cube_data_dict[network_side] = network_vols3d_subject[network_side][(x_i):(x_i + x_patch),
                                                           (y_i):(y_i + y_patch), (z_i):(z_i + z_patch)].astype(
                                np.float32)

                    if len(cube_data_dict) is len(self.dictionary.get_network_sides(group)):

                        if self.z_correction:
                            if self._Z_CORR:

                                cube_data_dict[NN_IN] = ((cube_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                                    np.float32)
                            else:
                                raise ValueError(
                                    'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                                    or they must be sat manually in the object')

                        if self.data_augmentation:
                            cube_data_dict = self._do_data_augmentation(group, cube_data_dict)

                        tfrecord_cube_list.append(cube_data_dict)

        return tfrecord_cube_list

    def run(self, subjects_buffer_size=1):

        if not isinstance(subjects_buffer_size, int):
            raise ValueError(' Error: "subjects_buffer_size" must be an integer.')

        groups_to_tfrecord = self.dictionary.get_list_groups()

        for group in groups_to_tfrecord:

            print('group %s' % group)

            writer = self._create_tfrecord_writer(group)
            subjects = self.dictionary.get_list_subjects(group)
            shuffle(subjects)
            for subject_id in range(0, len(subjects), subjects_buffer_size):

                subjects_buffer = subjects[subject_id:subject_id + subjects_buffer_size]
                list_slices_buffer = []
                gc.collect()
                for subject in subjects_buffer:
                    print('subject %s' % subject)

                    list_slices_buffer = list_slices_buffer + self._list_cubes_subject(group, subject)

                self._convert_to_serial_and_write(group, list_slices_buffer
                                                  , writer)
                list_slices_buffer = None
            self._print_info(group)


class Dataset_patch3d:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self.dst_path = ''
        self._shapes_tfrecord = dict()
        self.z_correction = False
        self.data_augmentation = False
        self.means_z = None
        self.stds_z = None
        self._used_roi_z_correction = False
        self.shuffle = False
        self.standar_maxmin = False
        self.network_side_label_check = None
        # self.stride_cube = 4
        self.group_stride_dict = None
        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'
        self.padding = None

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._DX = 5
        self._DY = 5
        self._ANGLE = 30

        self._read_function = None
        self._Z_CORR = False

    def _print_info(self, group):

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(os.path.join(self.dst_path, group + '.log'), 'w'))
        print = logger.info
        print('#######TFRECORD CREATION INFO#######################')
        print('-Dictionary of data: ')
        print(self.dictionary)
        print('-Group: %s' % group)
        print('-Shape tfrecord: %s' % self._shapes_tfrecord)
        print('-Z correction: %r' % self.z_correction)
        print('-Data augmentation: %r' % self.data_augmentation)
        print('-Means for Z correction: %s' % self.means_z)
        print('-Stds for Z correction: %s' % self.stds_z)
        print('-Use ROI in z correction: %r' % self._used_roi_z_correction)
        print('-Shuffle: %r' % self.shuffle)
        print('-NN_IN_DTYPE: %s' % self._NN_IN_DTYPE)
        print('-NN_OUT_DTYPE: %s' % self._NN_OUT_DTYPE)
        print('-NN_ROI_DTYPE: %s' % self._NN_ROI_DTYPE)
        print('DX for data aug: %d' % self._DX)
        print('DY for data aug: %d' % self._DY)
        print('ANGLE for data aug: %d' % self._ANGLE)
        print('Read function used: %s' % self._read_function)
        print('###################################################')
        logger.handlers.pop()

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol =  np.squeeze(img.get_data())

        return vol

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)

        if self.standar_maxmin:
            vol = (vol - vol.min()) / (vol.max() - vol.min())

        return vol

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))
                    if self.padding is not None:
                        vol = np.pad(vol, (
                        (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),
                                     'constant', constant_values=0)

                    vol = np.expand_dims(vol, axis=-1)

                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=-1)

        return network_vols3d_subject

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    def run(self):

        groups_to_tfrecord = self.dictionary.get_list_groups()

        dictionary_dataset = {}

        for group in groups_to_tfrecord:

            print('group %s' % group)

            subjects = self.dictionary.get_list_subjects(group)
            shuffle(subjects)

            group_subject_list = []

            for subject_id in subjects:
                dictionary_subject = {}

                subject_data = self._get_subject_data(group, subject_id)

                dictionary_subject['subject_data_dict'] = subject_data

                dictionary_subject['index_list'] = np.where(subject_data[NN_ROI][:, :, :, 0] == 1)

                group_subject_list.append(dictionary_subject)

            dictionary_dataset[group] = group_subject_list

        return dictionary_dataset


class ProductionCreator:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self.dst_path = ''
        self._shapes_tfrecord = dict()
        self.z_correction = False
        self.slice_dim = 2
        self.means_z = None
        self.stds_z = None
        self._used_roi_z_correction = False
        self.network_side_label_check = None
        self.slices_ba = 0
        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._read_function = None
        self._Z_CORR = False
        self.standar_maxmin = False

    def _print_info(self, group):

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(os.path.join(self.dst_path, group + '.log'), 'w'))
        print = logger.info
        print('#######PRODUCTION CREATION INFO#######################')
        print('-Dictionary of data: ')
        print(self.dictionary)
        print('-Group: %s' % group)
        print('-Shape tfrecord: %s' % self._shapes_tfrecord)
        print('-Z correction: %r' % self.z_correction)

        print('-Slice dimension: %d' % self.slice_dim)
        print('-Means for Z correction: %s' % self.means_z)
        print('-Stds for Z correction: %s' % self.stds_z)
        print('-Use ROI in z correction: %r' % self._used_roi_z_correction)
        print('-NN_IN_DTYPE: %s' % self._NN_IN_DTYPE)
        print('-NN_OUT_DTYPE: %s' % self._NN_OUT_DTYPE)
        print('-NN_ROI_DTYPE: %s' % self._NN_ROI_DTYPE)
        print('Read function used: %s' % self._read_function)
        print('###################################################')
        logger.handlers.pop()

    @staticmethod
    def _check_valid_output_slice(network_vols3d_subject, slice_id, network_side):

        slice_mask = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
            np.float32)

        if np.sum(slice_mask) > 0:
            contains_labels = True
        else:
            contains_labels = False
        return contains_labels

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol =  np.squeeze(img.get_data())

        return vol

    def resize_slices(self, new_size, group='all_groups', network_side=None):
        if group is 'all_groups':
            groups_to_do = self.dictionary.get_list_groups()
        else:
            groups_to_do = [group]

        if network_side is None:
            for group in groups_to_do:
                for network_side in self.dictionary.get_network_sides(group):
                    self._set_size_side(group, network_side, new_size)
        else:
            for group in groups_to_do:
                self._set_size_side(group, network_side, new_size)

    def _set_size_side(self, group, network_side, new_size):

        if not isinstance(new_size, tuple):
            raise ValueError('Error: "new_shape" must be a tuple')

        if len(new_size) != 2:
            raise ValueError('Error: "new_shape" must have two values')

        if network_side not in self.dictionary.get_network_sides(group):
            raise ValueError('Error: %s is not a network side')

        self._shapes_tfrecord[network_side] = new_size

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)
        # Always perform a np.rollaxis, we want the slicing position last
        if self.slice_dim != 2:
            vol = np.rollaxis(vol, self.slice_dim, 3)

        if self.standar_maxmin:
            vol = (vol - vol.min()) / (vol.max() - vol.min())

        if self.slices_ba != 0:
            vol = np.pad(vol, ((0, 0), (0, 0), (self.slices_ba, self.slices_ba)), 'edge')

        return vol

    @staticmethod
    def _resize_slice(slice_image, newsize, inter):
        if cv2_found:
            if inter is 'float':
                inter = cv2.INTER_CUBIC
            elif inter is 'int':
                inter = cv2.INTER_NEAREST

            slice_image = cv2.resize(slice_image, newsize,
                                     interpolation=inter)
            if slice_image.ndim is 2:
                slice_image = np.expand_dims(slice_image, axis=-1)

        else:
            raise ValueError(
                ' CV2 is not installed and is needed for resize slices, to install it use "sudo pip install opencv-python"')
        return slice_image

    def calc_z_correction(self, group, use_roi=False):

        # apriori the z_corr is only for network_in

        # INPUTS:
        # group: string that contains the group name that is going to be used to calculate the values for the z correction
        # use_roi: Boolean used to whether use a ROI to calculate the correction or not. If ROI channels != In channels
        # just the first roi channel is used
        # OUTPUT:Means and stds for z correction in a list in channel order

        means_per_channel = []
        stds_per_channel = []
        channel_list = self.dictionary.get_list_channels(group, NN_IN)
        subject_list = self.dictionary.get_list_subjects(group)
        for channel in channel_list:
            vol_list_flatten = []

            for subject in subject_list:

                vol_subject = self._read_data(self.dictionary.get_subject_path(group, NN_IN, channel, subject))
                # print(subject)
                # print(np.max(vol_subject))
                # print(np.min(vol_subject))

                if use_roi:
                    self._used_roi_z_correction = True
                    if len(self.dictionary.get_list_channels(group, NN_IN)) == len(
                            self.dictionary.get_list_channels(group, NN_ROI)):

                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, channel, subject))

                    else:
                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, 1, subject))

                    vol_list_flatten.append(np.extract(roi_subject, vol_subject))
                else:

                    vol_list_flatten.append(vol_subject.flatten())

            data_for_scale = np.concatenate(vol_list_flatten)

            means_per_channel.append(np.mean(data_for_scale))
            stds_per_channel.append(np.std(data_for_scale))

        self.means_z = means_per_channel
        self.stds_z = stds_per_channel

        self._Z_CORR = True
        print(means_per_channel)
        print(stds_per_channel)
        return means_per_channel, stds_per_channel

    def set_z_correction(self, means_z, stds_z):

        self.means_z = means_z

        self.stds_z = stds_z

        self._Z_CORR = True

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))

                    vol = np.expand_dims(vol, axis=-1)
                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=vol_channels_list[0].ndim - 1)
        return network_vols3d_subject

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    def _list_slices_subject(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        _, _, slices, _ = network_vols3d_subject[NN_IN].shape

        tfrecord_slice_list = []

        for slice_id in range(self.slices_ba, slices - self.slices_ba):

            if self.network_side_label_check:
                if not self._check_valid_output_slice(network_vols3d_subject, slice_id, self.network_side_label_check):
                    continue

            slice_data_dict = dict()

            for network_side in self.dictionary.get_network_sides(group):

                if network_side in list(self._shapes_tfrecord.keys()) and \
                        network_vols3d_subject[network_side][:, :,
                        (slice_id - self.slices_ba):(slice_id + self.slices_ba + 1), :].shape[0:2] != \
                        self._shapes_tfrecord[
                            network_side]:

                    slice_data_dict[network_side] = self._resize_slice(
                        network_vols3d_subject[network_side][:, :,
                        (slice_id - self.slices_ba):(slice_id + self.slices_ba + 1), :]
                        , self._shapes_tfrecord[network_side], self.dtypes_dict[network_side]).astype(np.float32)

                else:
                    slice_data_dict[network_side] = network_vols3d_subject[network_side][:, :,
                                                    (slice_id - self.slices_ba):(slice_id + self.slices_ba + 1),
                                                    :].astype(
                        np.float32)

                if slice_data_dict[network_side].ndim > 3:
                    slice_data_dict[network_side] = np.squeeze(slice_data_dict[network_side])

                if slice_data_dict[network_side].ndim == 2:
                    slice_data_dict[network_side] = np.expand_dims(slice_data_dict[network_side], axis=-1)

            if self.z_correction:
                if self._Z_CORR:

                    slice_data_dict[NN_IN] = ((slice_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                        np.float32)
                else:
                    raise ValueError(
                        'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                        or they must be sat manually in the object')

            tfrecord_slice_list.append(slice_data_dict)

        return tfrecord_slice_list

    def get_next_slice(self, group, subject, use_roi=True):

        print('group %s' % group)
        print('subject %s' % subject)
        #
        #
        # subjects = self.dictionary.get_list_subjects(group)
        # for subject in subjects:

        for slices in self._list_slices_subject(group, subject):
            if use_roi:
                yield (slices[NN_IN], slices[NN_OUT], slices[NN_ROI])
            else:
                yield (slices[NN_IN], slices[NN_OUT])

    def gen_nifti_from_list(self, slice_list, slice_gt_list, subject):

        array_slices = np.asarray(slice_list)
        array_slices = np.rollaxis(array_slices, 0, 3)

        img = nib.Nifti1Image(array_slices, np.eye(4))

        save_path = os.path.join(self.dst_path, subject + '.nii.gz')

        nib.save(img, save_path)

        slice_gt_list = np.asarray(slice_gt_list)
        slice_gt_list = np.rollaxis(slice_gt_list, 0, 3)

        img = nib.Nifti1Image(slice_gt_list, np.eye(4))

        save_path = os.path.join(self.dst_path, subject + '_gt.nii.gz')

        nib.save(img, save_path)


class TfrecordCreator3D:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self.dst_path = ''
        self._shapes_tfrecord = dict()
        self.z_correction = False
        self.data_augmentation = False
        self.slice_dim = 2
        self.means_z = None
        self.stds_z = None
        self._used_roi_z_correction = False
        self.shuffle = False
        self.network_side_label_check = None

        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._DX = 5
        self._DY = 5
        self._ANGLE = 360

        self._read_function = None
        self._Z_CORR = False

    def _print_info(self, group):

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(os.path.join(self.dst_path, group + '.log'), 'w'))
        print = logger.info
        print('#######TFRECORD CREATION INFO#######################')
        print('-Dictionary of data: ')
        print(self.dictionary)
        print('-Group: %s' % group)
        print('-Shape tfrecord: %s' % self._shapes_tfrecord)
        print('-Z correction: %r' % self.z_correction)
        print('-Data augmentation: %r' % self.data_augmentation)
        print('-Slice dimension: %d' % self.slice_dim)
        print('-Means for Z correction: %s' % self.means_z)
        print('-Stds for Z correction: %s' % self.stds_z)
        print('-Use ROI in z correction: %r' % self._used_roi_z_correction)
        print('-Shuffle: %r' % self.shuffle)
        print('-NN_IN_DTYPE: %s' % self._NN_IN_DTYPE)
        print('-NN_OUT_DTYPE: %s' % self._NN_OUT_DTYPE)
        print('-NN_ROI_DTYPE: %s' % self._NN_ROI_DTYPE)
        print('DX for data aug: %d' % self._DX)
        print('DY for data aug: %d' % self._DY)
        print('ANGLE for data aug: %d' % self._ANGLE)
        print('Read function used: %s' % self._read_function)
        print('###################################################')
        logger.handlers.pop()

    @staticmethod
    def _check_valid_output_slice(network_vols3d_subject, slice_id, network_side):

        slice_mask = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
            np.float32)

        if np.sum(slice_mask) > 0:
            contains_labels = True
        else:
            contains_labels = False
        return contains_labels

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol =  np.squeeze(img.get_data())

        return vol

    def resize_slices(self, new_size, group='all_groups', network_side=None):
        if group is 'all_groups':
            groups_to_do = self.dictionary.get_list_groups()
        else:
            groups_to_do = [group]

        if network_side is None:
            for group in groups_to_do:
                for network_side in self.dictionary.get_network_sides(group):
                    self._set_size_side(group, network_side, new_size)
        else:
            for group in groups_to_do:
                self._set_size_side(group, network_side, new_size)

    def _set_size_side(self, group, network_side, new_size):

        if not isinstance(new_size, tuple):
            raise ValueError('Error: "new_shape" must be a tuple')

        if len(new_size) != 2:
            raise ValueError('Error: "new_shape" must have two values')

        if network_side not in self.dictionary.get_network_sides(group):
            raise ValueError('Error: %s is not a network side')

        self._shapes_tfrecord[network_side] = new_size

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)
        # Always perform a np.rollaxis, we want the slicing position last
        if self.slice_dim != 2:
            vol = np.rollaxis(vol, self.slice_dim, 3)
        return vol

    @staticmethod
    def _resize_slice(slice_image, newsize, inter):
        if cv2_found:
            if inter is 'float':
                inter = cv2.INTER_CUBIC
            elif inter is 'int':
                inter = cv2.INTER_NEAREST

            slice_image = cv2.resize(slice_image, newsize,
                                     interpolation=inter)
            if slice_image.ndim is 2:
                slice_image = np.expand_dims(slice_image, axis=-1)

        else:
            raise ValueError(
                ' CV2 is not installed and is needed for resize slices, to install it use "sudo pip install opencv-python"')
        return slice_image

    def calc_z_correction(self, group, use_roi=False):

        # apriori the z_corr is only for network_in

        # INPUTS:
        # group: string that contains the group name that is going to be used to calculate the values for the z correction
        # use_roi: Boolean used to whether use a ROI to calculate the correction or not. If ROI channels != In channels
        # just the first roi channel is used
        # OUTPUT:Means and stds for z correction in a list in channel order

        means_per_channel = []
        stds_per_channel = []
        channel_list = self.dictionary.get_list_channels(group, NN_IN)
        subject_list = self.dictionary.get_list_subjects(group)
        for channel in channel_list:
            vol_list_flatten = []

            for subject in subject_list:

                vol_subject = self._read_data(self.dictionary.get_subject_path(group, NN_IN, channel, subject))
                if use_roi:
                    self._used_roi_z_correction = True
                    if len(self.dictionary.get_list_channels(group, NN_IN)) == len(
                            self.dictionary.get_list_channels(group, NN_ROI)):

                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, channel, subject))

                    else:
                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, 1, subject))

                    vol_list_flatten.append(np.extract(roi_subject, vol_subject))
                else:

                    vol_list_flatten.append(vol_subject.flatten())

            data_for_scale = np.concatenate(vol_list_flatten)

            means_per_channel.append(np.mean(data_for_scale))
            stds_per_channel.append(np.std(data_for_scale))

        self.means_z = means_per_channel
        self.stds_z = stds_per_channel

        self._Z_CORR = True

        return means_per_channel, stds_per_channel

    def set_z_correction(self, means_z, stds_z):

        self.means_z = means_z

        self.stds_z = stds_z

        self._Z_CORR = True

    def _create_tfrecord_writer(self, group):
        return tf.python_io.TFRecordWriter(os.path.join(self.dst_path, group + '.tfrecord'))

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))

                    vol = np.expand_dims(vol, axis=-1)
                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=vol_channels_list[0].ndim - 1)
        return network_vols3d_subject

    def _convert_to_serial_and_write(self, group, data_list, vols_list, slice_id_list, tfrecord_writer):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        num_examples = len(data_list)

        sample_indices = list(range(num_examples))
        if self.shuffle:
            random.shuffle(sample_indices)

        side_words = ['in', 'out', 'roi']

        for index in sample_indices:
            sample_features = dict()
            for network_side, shape_side_word in zip(self.dictionary.get_network_sides(group), side_words):
                sample_features[network_side] = _bytes_feature(data_list[index][network_side].tostring())
                rows, cols, channels = data_list[index][network_side].shape
                sample_features['rows_' + shape_side_word] = _int64_feature(rows)
                sample_features['cols_' + shape_side_word] = _int64_feature(cols)
                sample_features['channels_' + shape_side_word] = _int64_feature(channels)

            sample_features['vols'] = _bytes_feature(vols_list[index].tostring())
            sample_features['deep'] = _int64_feature(vols_list[index].shape[0])
            sample_features['slice_id'] = _int64_feature(slice_id_list[index])
            example = tf.train.Example(features=tf.train.Features(feature=sample_features))
            tfrecord_writer.write(example.SerializeToString())

    def _convert_to_serial_and_write_single(self, group, data, vol, slice_id, tfrecord_writer):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        side_words = ['in', 'out', 'roi']

        sample_features = dict()
        for network_side, shape_side_word in zip(self.dictionary.get_network_sides(group), side_words):
            sample_features[network_side] = _bytes_feature(data[network_side].tostring())
            rows, cols, channels = data[network_side].shape
            sample_features['rows_' + shape_side_word] = _int64_feature(rows)
            sample_features['cols_' + shape_side_word] = _int64_feature(cols)
            sample_features['channels_' + shape_side_word] = _int64_feature(channels)

        sample_features['vols'] = _bytes_feature(vol.tostring())
        sample_features['deep'] = _int64_feature(vol.shape[0])
        sample_features['slice_id'] = _int64_feature(slice_id)
        example = tf.train.Example(features=tf.train.Features(feature=sample_features))
        tfrecord_writer.write(example.SerializeToString())

    def set_data_augmentation(self, dx=5, dy=5, angle=360):

        self._DX = dx
        self._DY = dy
        self._ANGLE = angle

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    def _do_data_augmentation3d(self, group, slice_data_dict, vols_data):

        # dx = random.randrange(0, self._DX)
        # dy = random.randrange(0, self._DY)
        angle = random.randrange(0, self._ANGLE)

        # vols_data = interpolation.shift(vols_data, [0,dx, dy, 0],
        #                                                     mode='constant', cval=0)
        vols_data = interpolation.rotate(vols_data, angle, axes=(1, 2),
                                         reshape=False,
                                         mode='constant', cval=0)

        for network_side in self.dictionary.get_network_sides(group):

            dtype = self.dtypes_dict[network_side]

            if dtype is 'float':

                # slice_data_dict[network_side] = interpolation.shift(slice_data_dict[network_side], [dx, dy, 0],
                #                                                     mode='constant', cval=0)
                slice_data_dict[network_side] = interpolation.rotate(slice_data_dict[network_side], angle,
                                                                     reshape=False,
                                                                     mode='constant', cval=0)
            elif dtype is 'int':
                # slice_data_dict[network_side] = interpolation.shift(slice_data_dict[network_side], [dx, dy, 0],
                #                                                     mode='constant', cval=0, order=0)
                slice_data_dict[network_side] = interpolation.rotate(slice_data_dict[network_side], angle,
                                                                     reshape=False,
                                                                     mode='constant', cval=0, order=0)
            else:
                raise ValueError('Error: "dtype" in %s not recognized, %s' % (network_side, dtype))

        return slice_data_dict, vols_data

    @staticmethod
    def _resize_3d(vol_input):

        def resize_slice(slice_image, newsize, inter):

            if inter is 'float':
                inter = cv2.INTER_CUBIC
            elif inter is 'int':
                inter = cv2.INTER_NEAREST

            slice_image = cv2.resize(slice_image, newsize,
                                     interpolation=inter)

            return slice_image

        slice_list = []
        for slice_id in range(vol_input.shape[2]):
            slice = vol_input[:, :, slice_id, :]

            slice_resized = resize_slice(slice, (128, 128), 'float')

            slice_list.append(slice_resized)
        return np.asarray(slice_list)

    def _list_slices_subject(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        _, _, slices, _ = network_vols3d_subject[NN_IN].shape
        tfrecord_slice_list = []
        tfrecord_vols_list = []
        tfrecord_id_list = []
        if self._Z_CORR:
            network_vols3d_subject_zcorr = ((network_vols3d_subject[NN_IN] - self.means_z) / self.stds_z).astype(
                np.float32)

        network_vols3d_subject_zcorr_resize = self._resize_3d(network_vols3d_subject_zcorr)

        for slice_id in range(0, slices):

            if self.network_side_label_check:
                if not self._check_valid_output_slice(network_vols3d_subject, slice_id, self.network_side_label_check):
                    continue

            slice_data_dict = dict()

            for network_side in self.dictionary.get_network_sides(group):

                if network_side in list(self._shapes_tfrecord.keys()) and \
                        network_vols3d_subject[network_side][:, :, slice_id, :].shape[0:2] != self._shapes_tfrecord[
                    network_side]:

                    slice_data_dict[network_side] = self._resize_slice(
                        network_vols3d_subject[network_side][:, :, slice_id, :]
                        , self._shapes_tfrecord[network_side], self.dtypes_dict[network_side]).astype(np.float32)

                else:
                    slice_data_dict[network_side] = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
                        np.float32)

            if self.z_correction:
                if self._Z_CORR:

                    slice_data_dict[NN_IN] = ((slice_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                        np.float32)
                else:
                    raise ValueError(
                        'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                        or they must be sat manually in the object')

            if self.data_augmentation:
                slice_data_dict, network_vols3d_subject_zcorr_aug = self._do_data_augmentation3d(group, slice_data_dict,
                                                                                                 network_vols3d_subject_zcorr_resize)

                mask_slice = np.zeros(network_vols3d_subject_zcorr_aug[:, :, :, 0].shape, dtype=np.float32)
                mask_slice[slice_id, :, :] = 1
                mask_slice = np.expand_dims(mask_slice, axis=-1)
                network_vols3d_subject_zcorr_aug = np.concatenate((network_vols3d_subject_zcorr_aug, mask_slice),
                                                                  axis=3)

                tfrecord_id_list.append(slice_id)
                tfrecord_slice_list.append(slice_data_dict)
                tfrecord_vols_list.append(network_vols3d_subject_zcorr_aug)

            return tfrecord_slice_list, tfrecord_vols_list, tfrecord_id_list

    def _list_slices_subject_yield(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        _, _, slices, _ = network_vols3d_subject[NN_IN].shape

        if self._Z_CORR:
            network_vols3d_subject_zcorr = ((network_vols3d_subject[NN_IN] - self.means_z) / self.stds_z).astype(
                np.float32)

        network_vols3d_subject_zcorr_resize = self._resize_3d(network_vols3d_subject_zcorr)

        sample_indices = list(range(slices))
        if self.shuffle:
            random.shuffle(sample_indices)

        for slice_id in sample_indices:

            if self.network_side_label_check:
                if not self._check_valid_output_slice(network_vols3d_subject, slice_id, self.network_side_label_check):
                    continue

            slice_data_dict = dict()

            for network_side in self.dictionary.get_network_sides(group):

                if network_side in list(self._shapes_tfrecord.keys()) and \
                        network_vols3d_subject[network_side][:, :, slice_id, :].shape[0:2] != self._shapes_tfrecord[
                    network_side]:

                    slice_data_dict[network_side] = self._resize_slice(
                        network_vols3d_subject[network_side][:, :, slice_id, :]
                        , self._shapes_tfrecord[network_side], self.dtypes_dict[network_side]).astype(np.float32)

                else:
                    slice_data_dict[network_side] = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
                        np.float32)

            if self.z_correction:
                if self._Z_CORR:

                    slice_data_dict[NN_IN] = ((slice_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                        np.float32)
                else:
                    raise ValueError(
                        'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                        or they must be sat manually in the object')

            if self.data_augmentation:
                slice_data_dict, network_vols3d_subject_zcorr_aug = self._do_data_augmentation3d(group, slice_data_dict,
                                                                                                 network_vols3d_subject_zcorr_resize)

                mask_slice = np.zeros(network_vols3d_subject_zcorr_aug[:, :, :, 0].shape, dtype=np.float32)
                mask_slice[slice_id, :, :] = 1
                mask_slice = np.expand_dims(mask_slice, axis=-1)
                network_vols3d_subject_zcorr_aug = np.concatenate((network_vols3d_subject_zcorr_aug, mask_slice),
                                                                  axis=3)

            #     tfrecord_id_list.append(slice_id)
            #     tfrecord_slice_list.append(slice_data_dict)
            #     tfrecord_vols_list.append(network_vols3d_subject_zcorr_aug)
            #
            # return tfrecord_slice_list, tfrecord_vols_list, tfrecord_id_list

            yield network_vols3d_subject_zcorr_aug, slice_data_dict, slice_id

    def run(self, subjects_buffer_size=1):

        if not isinstance(subjects_buffer_size, int):
            raise ValueError(' Error: "subjects_buffer_size" must be an integer.')

        groups_to_tfrecord = self.dictionary.get_list_groups()

        for group in groups_to_tfrecord:

            print('group %s' % group)

            writer = self._create_tfrecord_writer(group)
            subjects = self.dictionary.get_list_subjects(group)
            for subject_id in range(0, len(subjects)):

                subject = subjects[subject_id]

                print('subject %s' % subject)
                # list_slices, list_vols, slice_id_list =  self._list_slices_subject(group, subject)

                for vol, slice, slice_id in self._list_slices_subject_yield(group, subject):
                    self._convert_to_serial_and_write_single(group, slice, vol, slice_id, writer)

            self._print_info(group)


class ProductionCreator_patch3d:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self.dst_path = ''
        self._shapes_tfrecord = dict()
        self.z_correction = False
        self.patch_shape = (32, 32, 32)
        self.stride_cube = 16
        self.means_z = None
        self.stds_z = None
        self._used_roi_z_correction = False
        self.network_side_label_check = None
        self.check_valid_mask = False

        self.valid_in_mask = (
        (int(self.patch_shape[0] / 2 - self.stride_cube / 2), int(self.patch_shape[0] / 2 + self.stride_cube / 2)),
        (int(self.patch_shape[1] / 2 - self.stride_cube / 2), int(self.patch_shape[1] / 2 + self.stride_cube / 2)),
        (int(self.patch_shape[2] / 2 - self.stride_cube / 2), int(self.patch_shape[2] / 2 + self.stride_cube / 2)))

        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._read_function = None
        self._Z_CORR = False
        self.standar_maxmin = False

    def _print_info(self, group):

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        logger.addHandler(logging.FileHandler(os.path.join(self.dst_path, group + '.log'), 'w'))
        print = logger.info
        print('#######PRODUCTION CREATION INFO#######################')
        print('-Dictionary of data: ')
        print(self.dictionary)
        print('-Group: %s' % group)
        print('-Shape tfrecord: %s' % self._shapes_tfrecord)
        print('-Z correction: %r' % self.z_correction)

        print('-Patch shape: %s' % self.patch_shape)
        print('-Means for Z correction: %s' % self.means_z)
        print('-Stds for Z correction: %s' % self.stds_z)
        print('-Use ROI in z correction: %r' % self._used_roi_z_correction)
        print('-NN_IN_DTYPE: %s' % self._NN_IN_DTYPE)
        print('-NN_OUT_DTYPE: %s' % self._NN_OUT_DTYPE)
        print('-NN_ROI_DTYPE: %s' % self._NN_ROI_DTYPE)
        print('Read function used: %s' % self._read_function)
        print('###################################################')
        logger.handlers.pop()

    @staticmethod
    def _check_valid_output_slice(network_vols3d_subject, slice_id, network_side):

        slice_mask = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
            np.float32)

        if np.sum(slice_mask) > 0:
            contains_labels = True
        else:
            contains_labels = False
        return contains_labels

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol =  np.squeeze(img.get_data())

        return vol
    @staticmethod
    def _read_nii_dic(subject_path):

        if medpy_found:
            img, dic = medpy.io.load(subject_path)
        else:

            img = nib.load(subject_path)
            dic = img.header

        return dic, img.affine


    def resize_slices(self, new_size, group='all_groups', network_side=None):
        if group is 'all_groups':
            groups_to_do = self.dictionary.get_list_groups()
        else:
            groups_to_do = [group]

        if network_side is None:
            for group in groups_to_do:
                for network_side in self.dictionary.get_network_sides(group):
                    self._set_size_side(group, network_side, new_size)
        else:
            for group in groups_to_do:
                self._set_size_side(group, network_side, new_size)

    def _set_size_side(self, group, network_side, new_size):

        if not isinstance(new_size, tuple):
            raise ValueError('Error: "new_shape" must be a tuple')

        if len(new_size) != 2:
            raise ValueError('Error: "new_shape" must have two values')

        if network_side not in self.dictionary.get_network_sides(group):
            raise ValueError('Error: %s is not a network side')

        self._shapes_tfrecord[network_side] = new_size

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)

        if self.standar_maxmin:
            vol = (vol - vol.min()) / (vol.max() - vol.min())

        return vol

    def calc_z_correction(self, group, use_roi=False):

        # apriori the z_corr is only for network_in

        # INPUTS:
        # group: string that contains the group name that is going to be used to calculate the values for the z correction
        # use_roi: Boolean used to whether use a ROI to calculate the correction or not. If ROI channels != In channels
        # just the first roi channel is used
        # OUTPUT:Means and stds for z correction in a list in channel order

        means_per_channel = []
        stds_per_channel = []
        channel_list = self.dictionary.get_list_channels(group, NN_IN)
        subject_list = self.dictionary.get_list_subjects(group)
        for channel in channel_list:
            vol_list_flatten = []

            for subject in subject_list:

                vol_subject = self._read_data(self.dictionary.get_subject_path(group, NN_IN, channel, subject))
                # print(subject)
                # print(np.max(vol_subject))
                # print(np.min(vol_subject))

                if use_roi:
                    self._used_roi_z_correction = True
                    if len(self.dictionary.get_list_channels(group, NN_IN)) == len(
                            self.dictionary.get_list_channels(group, NN_ROI)):

                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, channel, subject))

                    else:
                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, 1, subject))

                    vol_list_flatten.append(np.extract(roi_subject, vol_subject))
                else:

                    vol_list_flatten.append(vol_subject.flatten())

            data_for_scale = np.concatenate(vol_list_flatten)

            means_per_channel.append(np.mean(data_for_scale))
            stds_per_channel.append(np.std(data_for_scale))

        self.means_z = means_per_channel
        self.stds_z = stds_per_channel

        self._Z_CORR = True
        print(means_per_channel)
        print(stds_per_channel)
        return means_per_channel, stds_per_channel

    def set_z_correction(self, means_z, stds_z):

        self.means_z = means_z

        self.stds_z = stds_z

        self._Z_CORR = True

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))

                    vol = np.expand_dims(vol, axis=-1)
                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=vol_channels_list[0].ndim - 1)
        return network_vols3d_subject

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    def get_next_slice(self, group, subject, use_roi=True):

        print('group %s' % group)
        print('subject %s' % subject)
        #
        #
        # subjects = self.dictionary.get_list_subjects(group)
        # for subject in subjects:

        for cubes_dict, index_tupple, shape_tupple in self._list_cubes_subject(group, subject):
            if use_roi:
                yield (cubes_dict[NN_IN], cubes_dict[NN_OUT], cubes_dict[NN_ROI], index_tupple, shape_tupple)
            else:
                yield (cubes_dict[NN_IN], cubes_dict[NN_OUT], index_tupple, shape_tupple)

    def _list_cubes_subject(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        for network_side in self.dictionary.get_network_sides(group):
            network_vols3d_subject[network_side] = np.pad(network_vols3d_subject[network_side], (
            (self.patch_shape[0], self.patch_shape[0]), (self.patch_shape[1], self.patch_shape[1]),
            (self.patch_shape[2], self.patch_shape[2]), (0, 0)),
                                                          'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        x, y, z, _ = network_vols3d_subject[NN_IN].shape

        for x_i in range(0, x, int(self.stride_cube)):
            for y_i in range(0, y, int(self.stride_cube)):
                for z_i in range(0, z, int(self.stride_cube)):

                    cube_data_dict = dict()

                    for network_side in self.dictionary.get_network_sides(group):

                        x_patch, y_patch, z_patch = self.patch_shape

                        if list(network_vols3d_subject[network_side][(x_i):(x_i + x_patch), (y_i):(y_i + y_patch),
                                (z_i):(z_i + z_patch)].shape[0:3]) != self.patch_shape:
                            continue


                        else:

                            cube_data_dict[network_side] = network_vols3d_subject[network_side][(x_i):(x_i + x_patch),
                                                           (y_i):(y_i + y_patch), (z_i):(z_i + z_patch)].astype(
                                np.float32)

                    if len(cube_data_dict) is len(self.dictionary.get_network_sides(group)):

                        if self.check_valid_mask:

                            # print(np.sum(cube_data_dict[NN_ROI]))

                            if np.sum(cube_data_dict[NN_ROI][self.valid_in_mask[0][0]:self.valid_in_mask[0][1]
                                      , self.valid_in_mask[1][0]:self.valid_in_mask[1][1]
                                      , self.valid_in_mask[2][0]:self.valid_in_mask[2][1]]) == 0:
                                continue

                        if self.z_correction:
                            if self._Z_CORR:

                                cube_data_dict[NN_IN] = ((cube_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                                    np.float32)
                            else:
                                raise ValueError(
                                    'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                                    or they must be sat manually in the object')

                        yield (cube_data_dict, (x_i, y_i, z_i), (x, y, z))

    def fill_cube(self, cube, index, cube_out):

        if cube_out[index[0]:(index[0] + self.patch_shape[0]), index[1]:(index[1] + self.patch_shape[1]),
           index[2]:(index[2] + self.patch_shape[2])].shape == cube[:, :, :, 0].shape:
            cube_out[index[0]:(index[0] + self.patch_shape[0]), index[1]:(index[1] + self.patch_shape[1]),
            index[2]:(index[2] + self.patch_shape[2])] = cube[:, :, :, 0]

        return cube_out

    def create_nifti_from_cube(self, cube, subject, group = None,meta_datos = False):
        if meta_datos == False:

            img = nib.Nifti1Image(cube, np.eye(4))

        if meta_datos == True:
            dic,affine = self._read_nii_dic(self.dictionary.get_subject_path(group,NN_OUT,1,subject))
            img = nib.Nifti1Image(cube, affine,dic)

        save_path = os.path.join(self.dst_path, subject + '_pseudo.nii.gz')

        nib.save(img, save_path)

    def fill_center(self, cube, index, cube_out):

        if cube_out[index[0]:(index[0] + self.patch_shape[0]), index[1]:(index[1] + self.patch_shape[1]),
           index[2]:(index[2] + self.patch_shape[2])].shape == cube[:, :, :, 0].shape:
            cube_out[int(index[0] + self.patch_shape[0] / 2), int(index[1] + self.patch_shape[1] / 2), int(
                index[2] + self.patch_shape[2] / 2)] \
                = cube[int(self.patch_shape[0] / 2), int(self.patch_shape[1] / 2), int(self.patch_shape[2] / 2), 0]

            # print([int(index[0]+self.patch_shape[0]/2),int(index[1]+self.patch_shape[1]/2), int(index[2]+self.patch_shape[2]/2)])
            # print(cube[int(self.patch_shape[0]/2),int(self.patch_shape[1]/2),int(self.patch_shape[2]/2),0])

        return cube_out

    def fill_custom_cube(self, cube, index, cube_out, custom_shape=4):

        offset = list()

        offset.append(int(self.patch_shape[0] / 2 - custom_shape / 2))
        offset.append(int(self.patch_shape[1] / 2 - custom_shape / 2))
        offset.append(int(self.patch_shape[2] / 2 - custom_shape / 2))

        if cube_out[index[0]:(index[0] + self.patch_shape[0]), index[1]:(index[1] + self.patch_shape[1]),
           index[2]:(index[2] + self.patch_shape[2])].shape == cube[:, :, :, 0].shape:
            cube_out[(index[0] + offset[0]):(index[0] + offset[0] + custom_shape),
            (index[1] + offset[1]):(index[1] + offset[1] + custom_shape),
            (index[2] + offset[2]):(index[2] + offset[2] + custom_shape)] = \
                cube[offset[0]:(custom_shape + offset[0]), offset[1]:(custom_shape + offset[1]),
                offset[2]:(custom_shape + offset[2]), 0]

        return cube_out

    def correct_background(self, cube, group, subject):
        subject = self._get_subject_data( group, subject)
        vol_mask_int = np.asarray(subject[NN_ROI][:,:,:,0], np.int)

        cube[vol_mask_int == 0] = -1024
        return cube
class Dataset:
    def __init__(self, experiment_path, key_words_in, key_words_out, key_words_roi=None, group=None):

        self._raw_dictionary = self._read_experimentdict_from_folder(experiment_path, key_words_in, key_words_out,
                                                                     key_words_roi, group)
        self._dict_experiment = DataOperator(self._raw_dictionary)
        self._cross_validation = False
        self._cv_group = None
        self._cv_fold_size = None

    def __str__(self):
        return self._dict_experiment.__str__()

    @staticmethod
    def _create_dict_from_folder(path_subjects, key_words_in, key_words_out, key_words_roi):
        # This function create a dictionary used to feed a tfrecord creator used in a Deep Learning
        # experiment in Tensorflow.
        # path_subjects -> root folder that contains the subjects
        # key_words_* -> A list of keywords used to find the files used as input, output or roi in the
        # neural network

        subjects = list_nohidden_directories(path_subjects)

        channels = dict()
        channels[NN_IN] = len(key_words_in)
        channels[NN_OUT] = len(key_words_out)

        record_dict = dict()

        record_dict[NN_IN] = {}
        record_dict[NN_OUT] = {}

        key_words = dict()

        key_words[NN_IN] = key_words_in
        key_words[NN_OUT] = key_words_out
        if key_words_roi:
            record_dict[NN_ROI] = {}
            channels[NN_ROI] = len(key_words_roi)
            key_words[NN_ROI] = key_words_roi

        for subject in subjects:

            files = list_nohidden_files(os.path.join(path_subjects, subject))

            

            # lists to track the keys, every key must point one file and only one.
            track_keys = dict()
            track_keys[NN_IN] = key_words_in.copy()
            track_keys[NN_OUT] = key_words_out.copy()
            track_keys[NN_ROI] = key_words_roi.copy()

            for file in files:

                for network_side in record_dict.keys():

                    for key_word, channel in zip(key_words[network_side], range(1, channels[network_side] + 1)):

                        if key_word in file.split('/')[-1]:

                            if key_word in track_keys[network_side]:
                                track_keys[network_side].remove(key_word)
                            else:
                                raise ValueError(
                                    'ERROR: Key word "%s" was used in various files, each key must point an unique file.' % key_word)

                            record_dict[network_side].setdefault(channel, {}).update({os.path.basename(subject): file})

            error_keys = []
            for network_side in record_dict.keys():
                error_keys += track_keys[network_side]

            if error_keys:
                for key in error_keys:
                    print('ERROR: Key word "%s" was NOT used.' % key)

                raise ValueError(' ERROR: Unused keywords.')

        return record_dict

    def _read_experimentdict_from_folder(self, path_experiment, key_words_in, key_words_out, key_words_roi=None,
                                         group=None):
        # This function create a dictionary used to feed a tfrecord creator used in a Deep Learning
        # experiment in Tensorflow.
        # path_subjects -> root folder that contains the experiments.
        # key_words_* -> A list of keywords used to find the files used as input, output or roi in the
        # nerual network.

        if group == None:

            folders_experiment = list_nohidden_directories(path_experiment)

        else:
            if type(group) is list:
                folders_experiment = group
            else:
                folders_experiment = [group]

        experiment_dict = dict()

        for folder_experiment in folders_experiment:
            folder_dictionary = self._create_dict_from_folder(os.path.join(path_experiment, folder_experiment),
                                                              key_words_in,
                                                              key_words_out, key_words_roi)

            experiment_dict[folder_experiment] = folder_dictionary

        return experiment_dict

    def cv_data_iterator(self, group, fold_size, separable=None):
        self._cross_validation = True
        self._cv_group = group
        self._cv_fold_size = fold_size
        return self._iterator_crossvalidation(self._cv_group, self._cv_fold_size, separable)

    def get_dict_subjects(self, list, separable):

        dict_subjects = {}

        for subject in list:

            id = subject.split(separable)[0]
            if id in dict_subjects:
                dict_subjects[id].append(subject)
            else:
                dict_subjects[id] = []
                dict_subjects[id].append(subject)

        return dict_subjects

    def _iterator_crossvalidation(self, group, fold_size, separable):

        import collections

        subjects_dict = self.get_dict_subjects(self._dict_experiment.get_list_subjects(group), separable)

        subjects_rotate_list = collections.deque(list(subjects_dict.keys()))

        n_folds = np.int(np.ceil(np.float(len(subjects_rotate_list)) / fold_size))

        for _ in range(n_folds):

            dict_cv = dict()

            dict_cv[CV_TRAIN] = self._dict_experiment.get_groups_dictionary(group)[group]
            dict_cv[CV_TEST] = self._dict_experiment.get_groups_dictionary(group)[group]

            subjects_train = list(subjects_rotate_list)[fold_size::]
            subjects_test = list(subjects_rotate_list)[0:fold_size]

            subjects_rotate_list.rotate(-fold_size)

            for network_side in self._dict_experiment.get_network_sides(group):
                for channel in self._dict_experiment.get_list_channels(group, network_side):
                    for subject_name in subjects_rotate_list:

                        if subject_name not in subjects_train:
                            for subject_repeat in subjects_dict[subject_name]:
                                del (dict_cv[CV_TRAIN][network_side][channel][subject_repeat])

                        if subject_name not in subjects_test:
                            for subject_repeat in subjects_dict[subject_name]:
                                del (dict_cv[CV_TEST][network_side][channel][subject_repeat])

            yield (DataOperator(dict_cv))

    def get_groups_keys(self):
        return self._dict_experiment.get_list_groups()

    def get_subjects_keys(self, group):
        return self._dict_experiment.get_list_subjects(group)

    def get_data_from_groups(self, groups='all'):

        return self._dict_experiment.get_data(groups=groups)


class Experiment:
    def __init__(self, experiment_path, folder_records='records', folder_logs='logs_tb', folder_models='models',
                 folder_session_details='session_details', folder_production='production'):

        self.logger = None
        self.print = None
        self._data_set = None
        self.experiment_path = experiment_path
        self._continue_session = True
        self._folder_records = folder_records
        self._folder_logs = folder_logs
        self._folder_models = folder_models
        self._folder_session_details = folder_session_details
        self._folder_production = folder_production
        self._train_name = None
        self._test_name = None
        self.channels_input = None
        self.channels_output = None
        self._session_name = None

        self.dataset_info = None

        self.tfrecord_info = None

        self.train_info = None
        self._clean_up_session = False
        self._generate_folders()

    def _get_channels_info(self):
        channels = dict()
        for network_side in self._data_set.get_network_sides(self._train_name):
            channels[network_side] = len(self._data_set.get_list_channels(self._train_name, network_side))
        return channels

    def get_data(self):

        if self._test_name:

            return self._data_set.get_data([self._train_name, self._test_name])

        else:
            return self._data_set.get_data(self._train_name)

    def _generate_folders(self):

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        if not os.path.exists(os.path.join(self.experiment_path, self._folder_records)):
            os.makedirs(os.path.join(self.experiment_path, self._folder_records))
        if not os.path.exists(os.path.join(self.experiment_path, self._folder_logs)):
            os.makedirs(os.path.join(self.experiment_path, self._folder_logs))
        if not os.path.exists(os.path.join(self.experiment_path, self._folder_session_details)):
            os.makedirs(os.path.join(self.experiment_path, self._folder_session_details))
        if not os.path.exists(os.path.join(self.experiment_path, self._folder_models)):
            os.makedirs(os.path.join(self.experiment_path, self._folder_models))
        if not os.path.exists(os.path.join(self.experiment_path, self._folder_production)):
            os.makedirs(os.path.join(self.experiment_path, self._folder_production))

    def get_records_path(self):

        return os.path.join(self.experiment_path, self._folder_records)

    def get_log_session_path(self):

        return os.path.join(self.experiment_path, self._folder_logs, self._session_name)

    def get_logs_experiment(self):

        return os.path.join(self.experiment_path, self._folder_logs)

    def get_production_path(self):

        return os.path.join(self.experiment_path, self._folder_production)

    def get_models_session_path(self):

        return os.path.join(self.experiment_path, self._folder_models, self._session_name)

    def get_details_session_path(self):

        return os.path.join(self.experiment_path, self._folder_session_details, self._session_name)

    def get_production_session_path(self):

        return os.path.join(self.experiment_path, self._folder_production, self._session_name)

    def get_record_train_name(self):
        return self._train_name + '.tfrecord'

    def get_record_test_name(self):
        return self._test_name + '.tfrecord'

    def _generate_folders_session(self):

        if self._clean_up_session and not self._continue_session:
            print('Session cleaned up!')
            self._delete_session()

        if not os.path.exists(self.get_log_session_path()):

            os.makedirs(self.get_log_session_path())
        elif not self._continue_session and not self._clean_up_session:

            print(' session name %s already used.' % self._session_name)
            print(' ¿Are you retraining?, please confirm flags "session_continue" and "clean_up" ')
            raise ValueError('Set session_continue flag and clean_up flag')
        if not os.path.exists(self.get_models_session_path()):

            os.makedirs(self.get_models_session_path())
        elif not self._continue_session and not self._clean_up_session:

            print(' session name %s already used.' % self._session_name)
            print(' ¿Are you retraining?, please confirm flags "session_continue" and "clean_up" ')
            raise ValueError('Set session_continue flag and clean_up flag')
        if not os.path.exists(self.get_details_session_path()):

            os.makedirs(self.get_details_session_path())
        elif not self._continue_session and not self._clean_up_session:

            print(' session name %s already used.' % self._session_name)
            print(' ¿Are you retraining?, please confirm flags "session_continue" and "clean_up" ')
            raise ValueError('Set session_continue flag and clean_up flag')

        if not os.path.exists(self.get_production_session_path()):

            os.makedirs(self.get_production_session_path())
        elif not self._continue_session and not self._clean_up_session:

            print(' session name %s already used.' % self._session_name)
            print(' ¿Are you retraining?, please confirm flags "session_continue" and "clean_up" ')
            raise ValueError('Set session_continue flag and clean_up flag')

    def set_session(self, session_name, session_data, train_name, test_name=None, continue_session=True,
                    clean_up=False):

        assert isinstance(session_data, DataOperator)

        self._data_set = session_data

        self._session_name = session_name
        self._train_name = train_name
        if test_name:
            self._test_name = test_name
        self._continue_session = continue_session
        self._clean_up_session = clean_up

        self._generate_folders_session()
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger()
        channels_info = self._get_channels_info()
        self.channels_input = channels_info[NN_IN]
        self.channels_output = channels_info[NN_OUT]
        self.print_info()

    def set_record_session(self, session_data, train_name, test_name=None):

        assert isinstance(session_data, DataOperator)

        self._data_set = session_data

        self._train_name = train_name
        if test_name:
            self._test_name = test_name

    def print_info(self):
        print = self.open_logger()
        print('############################################################')
        print('-Experiment path: %s' % self.experiment_path)
        print('-Session name: %s' % self._session_name)
        print('-dataset used:')
        print(self._data_set)
        self.close_logger()

    def open_logger(self):
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.get_details_session_path(), self._session_name + '.log'), 'a'))
        return self.logger.info

    def close_logger(self):
        self.logger.handlers.pop()

    def _delete_session(self):

        import shutil
        if os.path.exists(os.path.join(self.get_models_session_path())):
            shutil.rmtree(os.path.join(self.get_models_session_path(), ''))
        if os.path.exists(os.path.join(self.get_log_session_path())):
            shutil.rmtree(os.path.join(self.get_log_session_path(), ''))
        if os.path.exists(os.path.join(self.get_details_session_path())):
            shutil.rmtree(os.path.join(self.get_details_session_path(), ''))

    def get_continue_flag(self):
        return self._continue_session


class InputDictCreator:

    def __init__(self, dictionary):

        assert (isinstance(dictionary, DataOperator))

        self.dictionary = dictionary
        self._shapes_tfrecord = dict()
        self.z_correction = False
        self.slice_dim = 2
        self.means_z = None
        self.stds_z = None

        self.network_side_label_check = None

        self._NN_IN_DTYPE = 'float'
        self._NN_OUT_DTYPE = 'float'
        self._NN_ROI_DTYPE = 'int'

        self.dtypes_dict = {
            NN_IN: self._NN_IN_DTYPE,
            NN_OUT: self._NN_OUT_DTYPE,
            NN_ROI: self._NN_ROI_DTYPE
        }

        self._read_function = None
        self._Z_CORR = False

    @staticmethod
    def _read_nii(subject_path):

        if medpy_found:
            vol, _ = medpy.io.load(subject_path)
        else:
            img = nib.load(subject_path)
            vol =  np.squeeze(img.get_data())

        return vol

    def resize_slices(self, new_size, group='all_groups', network_side=None):
        if group is 'all_groups':
            groups_to_do = self.dictionary.get_list_groups()
        else:
            groups_to_do = [group]

        if network_side is None:
            for group in groups_to_do:
                for network_side in self.dictionary.get_network_sides(group):
                    self._set_size_side(group, network_side, new_size)
        else:
            for group in groups_to_do:
                self._set_size_side(group, network_side, new_size)

    def _set_size_side(self, group, network_side, new_size):

        if not isinstance(new_size, tuple):
            raise ValueError('Error: "new_shape" must be a tuple')

        if len(new_size) != 2:
            raise ValueError('Error: "new_shape" must have two values')

        if network_side not in self.dictionary.get_network_sides(group):
            raise ValueError('Error: %s is not a network side')

        self._shapes_tfrecord[network_side] = new_size

    def set_read_function(self, new_read_function):
        self._read_function = new_read_function

    def _read_data(self, subject_path):

        if self._read_function is not None:

            vol = self._read_function(subject_path)
        else:
            vol = self._read_nii(subject_path)
        # Always perform a np.rollaxis, we want the slicing position last
        if self.slice_dim != 2:
            vol = np.rollaxis(vol, self.slice_dim, 3)
        return vol

    @staticmethod
    def _resize_slice(slice_image, newsize, inter):
        if cv2_found:
            if inter is 'float':
                inter = cv2.INTER_CUBIC
            elif inter is 'int':
                inter = cv2.INTER_NEAREST

            slice_image = cv2.resize(slice_image, newsize,
                                     interpolation=inter)
            if slice_image.ndim is 2:
                slice_image = np.expand_dims(slice_image, axis=-1)

        else:
            raise ValueError(
                ' CV2 is not installed and is needed for resize slices, to install it use "sudo pip install opencv-python"')
        return slice_image

    def calc_z_correction(self, group, use_roi=False):

        # apriori the z_corr is only for network_in

        # INPUTS:
        # group: string that contains the group name that is going to be used to calculate the values for the z correction
        # use_roi: Boolean used to whether use a ROI to calculate the correction or not. If ROI channels != In channels
        # just the first roi channel is used
        # OUTPUT:Means and stds for z correction in a list in channel order

        means_per_channel = []
        stds_per_channel = []
        channel_list = self.dictionary.get_list_channels(group, NN_IN)
        subject_list = self.dictionary.get_list_subjects(group)
        for channel in channel_list:
            vol_list_flatten = []

            for subject in subject_list:

                vol_subject = self._read_data(self.dictionary.get_subject_path(group, NN_IN, channel, subject))
                if use_roi:
                    if len(self.dictionary.get_list_channels(group, NN_IN)) == len(
                            self.dictionary.get_list_channels(group, NN_ROI)):

                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, channel, subject))

                    else:
                        roi_subject = self._read_data(
                            self.dictionary.get_subject_path(group, NN_ROI, 1, subject))

                    vol_list_flatten.append(np.extract(roi_subject, vol_subject))
                else:

                    vol_list_flatten.append(vol_subject.flatten())

            data_for_scale = np.concatenate(vol_list_flatten)

            means_per_channel.append(np.mean(data_for_scale))
            stds_per_channel.append(np.std(data_for_scale))

        self.means_z = means_per_channel
        self.stds_z = stds_per_channel

        self._Z_CORR = True

        return means_per_channel, stds_per_channel

    def set_z_correction(self, means_z, stds_z):

        self.means_z = means_z

        self.stds_z = stds_z

        self._Z_CORR = True

    def _get_subject_data(self, group, subject):
        network_vols3d_subject = {}
        for network_side in self.dictionary.get_network_sides(group):

            if self.dictionary.get_list_channels(group, network_side):

                vol_channels_list = []
                for channel in self.dictionary.get_list_channels(group, network_side):
                    vol = self._read_data(self.dictionary.get_subject_path(group, network_side, channel, subject))

                    vol = np.expand_dims(vol, axis=-1)
                    vol_channels_list.append(vol)

                network_vols3d_subject[network_side] = np.concatenate(vol_channels_list,
                                                                      axis=vol_channels_list[0].ndim - 1)
        return network_vols3d_subject

    def set_dtypes(self, dtype_in='float', dtype_out='float', dtype_roi='int'):

        if dtype_in != 'float' and dtype_in != 'int' or dtype_out != 'float' and dtype_out != 'int' or dtype_roi != 'float' and dtype_roi != 'int':
            raise ValueError(' Bad dtype founded.')

        self._NN_IN_DTYPE = dtype_in
        self._NN_OUT_DTYPE = dtype_out
        self._NN_ROI_DTYPE = dtype_roi
        for network_side, dtype in zip(self.dictionary.get_network_sides(self.dictionary.get_list_groups()[0]),
                                       [self._NN_IN_DTYPE, self._NN_OUT_DTYPE, self._NN_ROI_DTYPE]):
            self.dtypes_dict[network_side] = dtype

    @staticmethod
    def _check_valid_output_slice(network_vols3d_subject, slice_id, network_side):

        slice_mask = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
            np.float32)

        if np.sum(slice_mask) > 0:
            contains_labels = True
        else:
            contains_labels = False
        return contains_labels

    def _list_slices_subject(self, group, subject):

        network_vols3d_subject = self._get_subject_data(group, subject)

        _, _, slices, _ = network_vols3d_subject[NN_IN].shape
        tfrecord_slice_list = []

        for slice_id in range(0, slices):

            if self.network_side_label_check:
                if self._check_valid_output_slice(network_vols3d_subject, slice_id, self.network_side_label_check):
                    continue

            slice_data_dict = dict()

            for network_side in self.dictionary.get_network_sides(group):

                if network_side in list(self._shapes_tfrecord.keys()) and \
                        network_vols3d_subject[network_side][:, :, slice_id, :].shape[0:2] != self._shapes_tfrecord[
                    network_side]:

                    slice_data_dict[network_side] = self._resize_slice(
                        network_vols3d_subject[network_side][:, :, slice_id, :]
                        , self._shapes_tfrecord[network_side], self.dtypes_dict[network_side]).astype(np.float32)

                else:
                    slice_data_dict[network_side] = network_vols3d_subject[network_side][:, :, slice_id, :].astype(
                        np.float32)

            if self.z_correction:
                if self._Z_CORR:

                    slice_data_dict[NN_IN] = ((slice_data_dict[NN_IN] - self.means_z) / self.stds_z).astype(
                        np.float32)
                else:
                    raise ValueError(
                        'Error: The calculation of the Z correction input parameters must be done before creating the tfrecord, \
                        or they must be sat manually in the object')

            tfrecord_slice_list.append(slice_data_dict)

        return tfrecord_slice_list

    def create_dict_group(self, group):

        print('group %s' % group)

        subjects = self.dictionary.get_list_subjects(group)
        group_dictionary_iter = dict()
        for subject in subjects:
            group_dictionary_iter[subject] = []
            print('subject %s' % subject)

            group_dictionary_iter[subject] = self._list_slices_subject(group, subject)

        return group_dictionary_iter
