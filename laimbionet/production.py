import tensorflow as tf
import numpy as np
from laimbionet.input import Experiment
import os
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

class ProductionNetwork:
    def __init__(self):



        self.network = None
        self.args_network = None
        self._session = None
        self._saver = None
        self.use_roi = False
        self.model_to_load = None
        self.model_epoch = None
        self.chkpt_folder = None
        self._nodes_graph = dict()
        self._input_shape = None
        self._read_function = None


    def initialize(self,experiment):

        self.set_experiment(experiment)
        if self.model_epoch == None:
            self.chkpt_folder = self.models_path
        else:
            self.model_to_load = os.path.join(self.models_path, 'epoch_' + str(self.model_epoch) + ".ckpt")

        self.load_network()

    def set_experiment(self, experiment):

        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()





        print = self.experiment.open_logger()
        print('##Seting Experiment in testing##')
        print('-Models paths: %s' % self.models_path)


        self.experiment.close_logger()


    def load_network(self):
        self._session = tf.Session()


        inputs_tensor = tf.placeholder(tf.float32, [1,None,None,None], name='inputs_tensor')
        if self.use_roi:
            rois_tensor = tf.placeholder(tf.float32,[1,None,None,None], name='rois_tensor')

            self._nodes_graph['rois_tensor'] = rois_tensor



        self._nodes_graph['inputs_tensor'] = inputs_tensor

        graph_network = self.network(**self.args_network, input=inputs_tensor, is_training=False)

        if self.use_roi:
            outputs_tensor = graph_network.output* rois_tensor
        else:
            outputs_tensor = graph_network.output

        self._nodes_graph['outputs_tensor'] = outputs_tensor

        self._saver = tf.train.Saver()

        if self.model_to_load:
            self._saver.restore(self._session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        if self.chkpt_folder:
            latest_chkpt = tf.train.latest_checkpoint(self.chkpt_folder)
            self._saver.restore(self._session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)






    def produce_output_from_slice(self, inputs, rois = None):


        inputs = np.expand_dims(inputs, 0)


        if self.use_roi:
            rois = np.expand_dims(rois, 0)

            numpy_output = self._session.run(self._nodes_graph['outputs_tensor'],
                                            feed_dict={self._nodes_graph['inputs_tensor']: inputs, self._nodes_graph['rois_tensor']: rois})

        else:
            numpy_output = self._session.run(self._nodes_graph['outputs_tensor'],
                                            feed_dict={self._nodes_graph['inputs_tensor']: inputs})

        return np.squeeze(numpy_output,axis = 0)



    def close_network(self):

        self._session.close()

        tf.reset_default_graph()




class ProductionNetwork_patch3d:
    def __init__(self,channels = 1, patch_shape = (32,32,32)):



        self.network = None
        self.args_network = None
        self._session = None
        self._saver = None
        self.use_roi = False
        self.model_to_load = None
        self.model_epoch = None
        self.chkpt_folder = None
        self._nodes_graph = dict()
        self._input_shape = None
        self._read_function = None
        self.patch_shape = patch_shape
        self.channels = channels


    def initialize(self,experiment):

        self.set_experiment(experiment)
        if self.model_epoch == None:
            self.chkpt_folder = self.models_path
        else:
            self.model_to_load = os.path.join(self.models_path, 'epoch_' + str(self.model_epoch) + ".ckpt")

        self.load_network()

    def set_experiment(self, experiment):

        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()
        self.channels_input = experiment.channels_input
        self.channels_output = experiment.channels_output



        print = self.experiment.open_logger()
        print('##Seting Experiment in testing##')
        print('-Models paths: %s' % self.models_path)


        self.experiment.close_logger()


    def load_network(self):
        self._session = tf.Session()


        inputs_tensor = tf.placeholder(tf.float32, [1,self.patch_shape[0],self.patch_shape[1],self.patch_shape[2],self.channels_input], name='inputs_tensor')
        if self.use_roi:
            rois_tensor = tf.placeholder(tf.float32,[1,self.patch_shape[0],self.patch_shape[1],self.patch_shape[2],1], name='rois_tensor')

            self._nodes_graph['rois_tensor'] = rois_tensor



        self._nodes_graph['inputs_tensor'] = inputs_tensor

        graph_network = self.network(**self.args_network, input=inputs_tensor, is_training=False)

        if self.use_roi:
            outputs_tensor = graph_network.output* rois_tensor
        else:
            outputs_tensor = graph_network.output

        self._nodes_graph['outputs_tensor'] = outputs_tensor

        self._saver = tf.train.Saver()

        if self.model_to_load:
            self._saver.restore(self._session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        if self.chkpt_folder:
            latest_chkpt = tf.train.latest_checkpoint(self.chkpt_folder)
            self._saver.restore(self._session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)






    def produce_output_from_patch3d(self, inputs, rois = None):


        inputs = np.expand_dims(inputs, 0)


        if self.use_roi:
            rois = np.expand_dims(rois, 0)

            numpy_output = self._session.run(self._nodes_graph['outputs_tensor'],
                                            feed_dict={self._nodes_graph['inputs_tensor']: inputs, self._nodes_graph['rois_tensor']: rois})

        else:
            numpy_output = self._session.run(self._nodes_graph['outputs_tensor'],
                                            feed_dict={self._nodes_graph['inputs_tensor']: inputs})



        return np.squeeze(numpy_output,axis = 0)



    def close_network(self):

        self._session.close()

        tf.reset_default_graph()

