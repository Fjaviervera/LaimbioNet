import os
import tensorflow as tf
import time
from laimbionet.input import Experiment
import numpy as np
import matplotlib.pyplot as plt

class ClassifierTrainer:
    def __init__(self):

        self.epochs = None
        self.network = None
        self.args_network = None
        self.test = False
        self.learning_rate = None
        self.models_path = os.path.join(os.getcwd(), 'models')
        self.tfrecord_train = os.path.join(os.getcwd(), 'records', 'train.tfrecord')
        self.tfrecord_test = os.path.join(os.getcwd(), 'records', 'test.tfrecord')
        self.logs_path = os.path.join(os.getcwd(), 'logs_tb', 'test.tfrecord')
        self.max_to_models_tokeep = 5
        self.optimizer = tf.train.AdamOptimizer
        self.error_function = tf.losses.softmax_cross_entropy
        self.session = None
        self.saver = None
        self.use_roi = None
        self.model_to_load = None
        self.batch_size = 32
        self.nodes_graph = dict()
        self.shuffle_buffer = None
        self.continue_session = True
        self.val_size = None
        self.experiment = None
        self.channels_input = None
        self.classes_output = None
        self.loss_function = 'cross_entropy'
        self.max_imags_tensorboard = 3
        self.one_hot = False
        self.labels = None
        self.weight_roi = 0

    def set_experiment(self, experiment):
        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()
        self.channels_input = experiment.channels_input
        self.tfrecord_train = os.path.join(experiment.get_records_path(), experiment.get_record_train_name())
        self.continue_session = experiment.get_continue_flag()
        if self.test:
            self.tfrecord_test = os.path.join(experiment.get_records_path(), experiment.get_record_test_name())
        self.logs_path = experiment.get_log_session_path()

        print = self.experiment.open_logger()
        print('##Seting Experiment in trainer##')
        print('-Models paths: %s' % self.models_path)
        print('-Tfrecord Train: %s' % self.tfrecord_train)
        if self.test:
            print('-Tfrecord Test: %s' % self.tfrecord_test)
        else:
            print('-Tfrecord Test: False')
        print('-Logs path: %s' % self.logs_path)
        print('-Continue training: %r' % self.continue_session)

        self.experiment.close_logger()


    @staticmethod
    def gen_summaries_from_dicts(scalars_dict, images_dict):

        def create_scalar_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.scalar(name=name, tensor=tensor))

            return summary

        def create_image_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.image(name=name, tensor=tensor))

            return summary

        summaries = tf.summary.merge(
            create_scalar_summary(scalars_dict) + create_image_summary(images_dict))

        return summaries

    def _create_graph(self):
        self.session = tf.Session()


        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        tfrecord_path = tf.placeholder(tf.string, name='tfrecord_path')

        tfrecord_data = get_input_from_record(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                              val_size=self.val_size, buffer_shuffle=self.shuffle_buffer, one_hot=self.one_hot, labels = self.labels)
        if self.test:
            tfrecord_data_test = get_input_from_record(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                                       buffer_shuffle=self.shuffle_buffer, one_hot=self.one_hot, labels = self.labels)

        if self.use_roi:

            if self.val_size:

                inputs_batch, labels_batch, rois_batch, iterator, \
                inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test

            else:
                inputs_batch, labels_batch, rois_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test
        else:

            if self.val_size:
                inputs_batch, labels_batch, iterator, \
                inputs_batch_val, labels_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
            else:
                inputs_batch, labels_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test

        self.args_network['input_channels'] = self.channels_input
        self.args_network['output_channels'] = self.classes_output



        graph_network = self.network(**self.args_network, input=inputs_batch, is_training=True)
        net_outputs_batch = graph_network.output
        if self.use_roi and self.weight_roi != 0:
            rois_batch_weigh = rois_batch + (labels_batch * self.weight_roi)
            if self.val_size:
                rois_batch_val_weigh = rois_batch_val+ (labels_batch_val*self.weight_roi)

            if self.test:
                rois_batch_test_weigh = rois_batch_test+ (labels_batch_test* self.weight_roi)


        if self.loss_function is 'cross_entropy':
            labels_batch = tf.cast(labels_batch, dtype=tf.uint8)


        if self.test:
            graph_network_test = self.network(**self.args_network, input=inputs_batch_test, reuse=True,
                                              is_training=False)
            net_outputs_batch_test = graph_network_test.output
            if self.loss_function is 'cross_entropy':
                labels_batch_test = tf.cast(labels_batch_test, dtype=tf.uint8)

        if self.val_size:
            graph_network_val = self.network(**self.args_network, reuse=True, input=inputs_batch_val, is_training=False)
            net_outputs_batch_val = graph_network_val.output
            if self.loss_function is 'cross_entropy':
                labels_batch_val = tf.cast(labels_batch_val, dtype=tf.uint8)



        labels_batch_onehot = tf.one_hot(tf.cast(labels_batch, dtype=tf.uint8)[:, :, :, 0], self.classes_output)
        if self.val_size:
            labels_batch_onehot_val = tf.one_hot(tf.cast(labels_batch_val, dtype=tf.uint8)[:, :, :, 0],
                                                 self.classes_output)
        if self.test:
            labels_batch_onehot_test = tf.one_hot(tf.cast(labels_batch_test, dtype=tf.uint8)[:, :, :, 0],
                                                  self.classes_output)

        if self.loss_function is 'cross_entropy':
            if self.use_roi:
                if self.val_size:



                    loss_batch = self.error_function(labels_batch_onehot
                                                     , net_outputs_batch, rois_batch_weigh[:,:,:,0])
                    loss_batch_val = self.error_function(labels_batch_onehot_val
                                                         , net_outputs_batch_val, rois_batch_val_weigh[:,:,:,0])
                    if self.test:


                        loss_batch_test = self.error_function(labels_batch_onehot_test
                                                              , net_outputs_batch_test, rois_batch_test_weigh[:,:,:,0] )
                else:
                    loss_batch = self.error_function(labels_batch_onehot
                                                     , net_outputs_batch, rois_batch[:,:,:,0] )
                    if self.test:


                        loss_batch_test = self.error_function(labels_batch_onehot_test
                                                              , net_outputs_batch_test, rois_batch_test[:,:,:,0] )
            else:
                if self.val_size:



                    loss_batch = self.error_function(labels_batch_onehot
                                                     , net_outputs_batch)
                    loss_batch_val = self.error_function(labels_batch_onehot_val
                                                         , net_outputs_batch_val)
                    if self.test:
                        loss_batch_test = self.error_function(tf.one_hot( tf.cast(labels_batch_test, dtype=tf.uint8)[:,:,:,0], self.classes_output)
                                                              , net_outputs_batch_test)

                else:
                    loss_batch = self.error_function(tf.one_hot( tf.cast(labels_batch, dtype=tf.uint8)[:,:,:,0], self.classes_output)
                                                     , net_outputs_batch)
                    if self.test:


                        loss_batch_test = self.error_function(labels_batch_onehot_test
                                                              , net_outputs_batch_test)

        if self.loss_function is 'dice':
            if self.val_size:



                loss_batch =1 - dice_coe ( tf.nn.softmax(net_outputs_batch), labels_batch)
                loss_batch_val =1 - dice_coe (tf.nn.softmax(net_outputs_batch_val) , labels_batch_val)
                if self.test:
                    loss_batch_test = 1 - dice_coe (tf.nn.softmax(net_outputs_batch_test), labels_batch_test)

            else:
                loss_batch =1 - dice_coe ( tf.nn.softmax(net_outputs_batch), labels_batch)
                if self.test:
                    loss_batch_test  =1- dice_coe(tf.nn.softmax(net_outputs_batch_test), labels_batch_test)




        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)
        update_epoch = epoch.assign_add(1)
        self.nodes_graph['global_step'] = global_step
        self.nodes_graph['epoch'] = epoch
        self.nodes_graph['update_epoch'] = update_epoch
        # Node to update epoch

        with tf.control_dependencies(update_ops):
            train_step = self.optimizer(self.learning_rate).minimize(loss_batch, global_step=global_step)

        self.saver = tf.train.Saver(max_to_keep=self.max_to_models_tokeep)

        if self.model_to_load:
            self.saver.restore(self.session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        elif self.continue_session:
            latest_chkpt = tf.train.latest_checkpoint(self.models_path)
            self.saver.restore(self.session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)

        # METRICS########################################33
        metrics_update_train = []
        metrics_train = dict()
        class_net_outputs_batch = tf.one_hot(tf.argmax(net_outputs_batch,3),self.classes_output)
        for class_id in range(self.classes_output):
            metrics_train['tp_' + str(class_id)], \
            tp_update = tf.metrics.true_positives(
                labels_batch_onehot[:, :, :, class_id], class_net_outputs_batch[:, :, :, class_id])

            metrics_update_train.append(tp_update)

            metrics_train['fp_' + str(class_id)], \
            fp_update = tf.metrics.false_positives(
                labels_batch_onehot[:, :, :, class_id], class_net_outputs_batch[:, :, :, class_id])

            metrics_update_train.append(fp_update)


            metrics_train['tn_' + str(class_id)], \
            tn_update = tf.metrics.true_negatives(
                labels_batch_onehot[:, :, :, class_id], class_net_outputs_batch[:, :, :, class_id])

            metrics_update_train.append(tn_update)

            metrics_train['fn_' + str(class_id)], \
            fn_update = tf.metrics.false_negatives(
                labels_batch_onehot[:, :, :, class_id], class_net_outputs_batch[:, :, :, class_id])

            metrics_update_train.append(fn_update)




        self.nodes_graph['metrics_update_train'] = metrics_update_train

        if self.val_size:

            class_net_outputs_batch_val = tf.one_hot(tf.argmax(net_outputs_batch_val,3), self.classes_output)

            metrics_update_val = []
            metrics_val = dict()

            for class_id in range(self.classes_output):
                metrics_val['tp_val_' + str(class_id)], \
                tp_update = tf.metrics.true_positives(
                    labels_batch_onehot_val[:, :, :, class_id], class_net_outputs_batch_val[:, :, :, class_id])

                metrics_update_val.append(tp_update)

                metrics_val['fp_val_' + str(class_id)], \
                fp_update = tf.metrics.false_positives(
                    labels_batch_onehot_val[:, :, :, class_id], class_net_outputs_batch_val[:, :, :, class_id])

                metrics_update_val.append(fp_update)

                metrics_val['tn_val_' + str(class_id)], \
                tn_update = tf.metrics.true_negatives(
                    labels_batch_onehot_val[:, :, :, class_id], class_net_outputs_batch_val[:, :, :, class_id])

                metrics_update_val.append(tn_update)

                metrics_val['fn_val_' + str(class_id)], \
                fn_update = tf.metrics.false_negatives(
                    labels_batch_onehot_val[:, :, :, class_id], class_net_outputs_batch_val[:, :, :, class_id])

                metrics_update_val.append(fn_update)




            self.nodes_graph['metrics_update_val'] = metrics_update_val



        if self.test:
            metrics_update_test = []
            metrics_test = dict()
            class_net_outputs_batch_test = tf.one_hot(tf.argmax(net_outputs_batch_test,3), self.classes_output)
            for class_id in range(self.classes_output):
                metrics_test['tp_test_' + str(class_id)], \
                tp_update = tf.metrics.true_positives(
                    labels_batch_onehot_test[:, :, :, class_id], class_net_outputs_batch_test[:, :, :, class_id])

                metrics_update_test.append(tp_update)

                metrics_test['fp_test_' + str(class_id)], \
                fp_update = tf.metrics.false_positives(
                    labels_batch_onehot_test[:, :, :, class_id], class_net_outputs_batch_test[:, :, :, class_id])

                metrics_update_test.append(fp_update)

                metrics_test['tn_test_' + str(class_id)], \
                tn_update = tf.metrics.true_negatives(
                    labels_batch_onehot_test[:, :, :, class_id], class_net_outputs_batch_test[:, :, :, class_id])

                metrics_update_test.append(tn_update)

                metrics_test['fn_test_' + str(class_id)], \
                fn_update = tf.metrics.false_negatives(
                    labels_batch_onehot_test[:, :, :, class_id], class_net_outputs_batch_test[:, :, :, class_id])

                metrics_update_test.append(fn_update)

            self.nodes_graph['metrics_update_test'] = metrics_update_test

        #######################################################################################

        ##SUMMARIES
        net_outputs__imag_batch = tf.cast(tf.argmax(net_outputs_batch,3), dtype=tf.float32)
        dict_scalars_train =dict()
        dict_scalars_train['loss'] =  loss_batch

        # dict_scalars_train = {**dict_scalars_train,**metrics_train}


        dict_images_train = dict()
        for channel_id in range(self.channels_input):
            dict_images_train['input ' + str(channel_id)] = tf.expand_dims( inputs_batch[0:self.max_imags_tensorboard,:,:,0], -1)


        dict_images_train['labels'] = tf.expand_dims( labels_batch[0:self.max_imags_tensorboard,:,:,0] * 255, -1)
        dict_images_train['outputs'] = tf.cast(tf.expand_dims( net_outputs__imag_batch[0:self.max_imags_tensorboard,:,:]  * rois_batch[0:3,:,:,0] * 255, -1), dtype=tf.uint8)




        train_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_train, dict_images_train)

        self.nodes_graph['train_summaries'] = train_summaries_worker

        if self.val_size:
            net_outputs__imag_batch_val =  tf.cast(tf.argmax(net_outputs_batch_val,3), dtype=tf.float32)

            dict_images_val = dict()
            for channel_id in range(self.channels_input):
                dict_images_val['input_val' + str(channel_id)] = tf.expand_dims( inputs_batch_val[0:self.max_imags_tensorboard,:,:,channel_id], -1)


            dict_images_val['labels_val'] =  tf.expand_dims( labels_batch_val[0:self.max_imags_tensorboard,:,:,0] * 255, -1)
            dict_images_val['outputs_val'] = tf.cast(tf.expand_dims( net_outputs__imag_batch_val[0:self.max_imags_tensorboard,:,:] * rois_batch_val[0:3,:,:,0] * 255, -1), dtype=tf.uint8)




            dict_scalars_val = dict()
            dict_scalars_val['loss_val'] = loss_batch_val
            # dict_scalars_val = {**dict_scalars_val, **metrics_val}


            val_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_val, dict_images_val)
            self.nodes_graph['val_summaries'] = val_summaries_worker

        self.nodes_graph['metrics_update_train'] = metrics_update_train





        self.nodes_graph['input_batch'] = inputs_batch
        self.nodes_graph['labels_batch'] = labels_batch
        self.nodes_graph['output_batch'] = net_outputs_batch
        self.nodes_graph['metrics_train'] = metrics_train
        self.nodes_graph['iterator'] = iterator
        if self.val_size:
            self.nodes_graph['metrics_update_val'] = metrics_update_val
            self.nodes_graph['input_batch_val'] = inputs_batch_val
            self.nodes_graph['labels_batch_val'] = labels_batch_val
            self.nodes_graph['iterator_val'] = iterator_val
            self.nodes_graph['output_batch_val'] = net_outputs_batch_val
            self.nodes_graph['loss_batch_val'] = loss_batch_val
            self.nodes_graph['metrics_val'] = metrics_val
        self.nodes_graph['tfrecord_path'] = tfrecord_path
        self.nodes_graph['is_training'] = is_training
        self.nodes_graph['learning_rate'] = learning_rate
        self.nodes_graph['train_step'] = train_step
        self.nodes_graph['loss_batch'] = loss_batch
        if self.test:
            self.nodes_graph['metrics_update_test'] = metrics_update_test
            self.nodes_graph['input_batch_test'] = inputs_batch_test
            self.nodes_graph['labels_batch_test'] = labels_batch_test
            self.nodes_graph['iterator_test'] = iterator_test
            self.nodes_graph['output_batch_test'] = net_outputs_batch_test
            self.nodes_graph['loss_batch_test'] = loss_batch_test
            self.nodes_graph['metrics_test'] = metrics_test

    def _log_setup(self):
        print = self.experiment.open_logger()

        print('###Training Setup###')
        print('-Args Network: ')
        print(self.args_network)
        print('-Network details: ')
        print(self.network)
        print('-Optimizer: ')
        print(self.optimizer)
        print('-Error function: ')
        print(self.error_function)
        print('-Use ROI: %r' % self.use_roi)
        print('-Model to load: %s' % self.model_to_load)
        print('-Batch size: %d' % self.batch_size)
        print('-Shuffle Buffer: %s' % self.shuffle_buffer)
        print('-Validation size: %s' % self.val_size)
        print('-Max to models to keep: %d' % self.max_to_models_tokeep)

        self.experiment.close_logger()
    def run(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING TRAINING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------

        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
        # summaries_iter = 1

        while keep_training:
            self.session.run(tf.local_variables_initializer())
            self.session.run(self.nodes_graph['iterator'].initializer,
                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            if self.val_size:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            # if self.test:
            #     self.session.run(self.nodes_graph['iterator_test'].initializer,
            #                      feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})

            print(" epoch number " + str(current_epoch))
            while True:

                try:

                    if self.val_size:
                        _, loss, loss_val, train_summaries, val_summaries, step,_,_,metrics_train_py,metrics_val_py = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['loss_batch_val'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['val_summaries'],
                             self.nodes_graph['global_step'], self.nodes_graph['metrics_update_train'],
                             self.nodes_graph['metrics_update_val'],self.nodes_graph['metrics_train'],self.nodes_graph['metrics_val']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})


                        summary_writer.add_summary(train_summaries, step)
                        summary_writer.add_summary(val_summaries, step)
                    else:
                        _, loss, summaries, step,_,metrics_train_py = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['global_step'], self.nodes_graph['metrics_update_train'],
                             self.nodes_graph['metrics_train']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})

                        summary_writer.add_summary(summaries, step)



                except tf.errors.OutOfRangeError:

                    summary = tf.Summary()

                    dice_lessions_train = 2 *metrics_train_py['tp_1']/( 2* metrics_train_py['tp_1']+metrics_train_py['fp_1'] +metrics_train_py['fn_1'])
                    # print(dice_lessions_train)
                    summary.value.add(tag=' dice_train', simple_value=dice_lessions_train)
                    summary_writer.add_summary(summary, current_epoch)

                    # for key, value in metrics_train_py.items():
                    #
                    #
                    #
                    #     summary.value.add(tag=key, simple_value=value)
                    # summary_writer.add_summary(summary, current_epoch)


                    if self.val_size:
                        # for key, value in metrics_val_py.items():
                        #     summary.value.add(tag=key, simple_value=value)
                        # summary_writer.add_summary(summary, current_epoch)

                        dice_lessions_val = 2 *metrics_val_py['tp_val_1']/( 2* metrics_val_py['tp_val_1']+metrics_val_py['fp_val_1'] +metrics_val_py['fn_val_1'])
                        # print(dice_lessions_val)
                        summary.value.add(tag=' dice_val', simple_value=dice_lessions_val)
                        summary_writer.add_summary(summary, current_epoch)

                    model_dump_path = os.path.join(self.models_path, 'epoch_' + str(current_epoch) + ".ckpt")
                    save_path = self.saver.save(self.session, model_dump_path)
                    print("Model saved in file: %s" % save_path)


                    print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                    print(
                        " -----------------------------------EPOCH FINISHED-----------------------------------------------------------------")
                    if self.test:

                        self.session.run(self.nodes_graph['iterator_test'].initializer,
                                         feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})

                        print(" epoch testing " + str(current_epoch))

                        loss_val_list = []

                        while True:

                            try:

                                loss_test,_, metrics_test_py = self.session.run(
                                    [self.nodes_graph['loss_batch_test'], self.nodes_graph['metrics_update_test']
                                     , self.nodes_graph['metrics_test']])

                                loss_val_list.append(loss_test)


                            except tf.errors.OutOfRangeError:

                                test_loss_mean = np.mean(loss_val_list)
                                summary = tf.Summary()
                                summary.value.add(tag="avg_test_loss_epoch", simple_value=test_loss_mean)

                                dice_lessions_test = 2 * metrics_test_py['tp_test_1'] / (
                                            2 * metrics_test_py['tp_test_1'] + metrics_test_py['fp_test_1'] + metrics_test_py[
                                        'fn_test_1'])
                                # print(dice_lessions_test)
                                summary.value.add(tag=' dice_test', simple_value=dice_lessions_test)
                                summary_writer.add_summary(summary, current_epoch)


                                # for key, value in metrics_test_py.items():
                                #     summary.value.add(tag=key , simple_value=value)
                                # summary_writer.add_summary(summary, current_epoch)

                                print('Test Loss: %f' % test_loss_mean)
                                print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                                print(
                                    " -----------------------------------EPOCH testing FINISHED-----------------------------------------------------------------")

                                break

                    # current_epoch += 1
                    _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
                    if self.epochs is not None:
                        if current_epoch > self.epochs:
                            keep_training = False

                    break

        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()


class RegressionTrainer:
    def __init__(self):

        self.epochs = None
        self.network = None
        self.args_network = None
        self.test = False
        self.learning_rate = None
        self.models_path = os.path.join(os.getcwd(), 'models')
        self.tfrecord_train = os.path.join(os.getcwd(), 'records', 'train.tfrecord')
        self.tfrecord_test = os.path.join(os.getcwd(), 'records', 'test.tfrecord')
        self.logs_path = os.path.join(os.getcwd(), 'logs_tb', 'test.tfrecord')
        self.max_to_models_tokeep = 5
        self.optimizer = tf.train.AdamOptimizer
        self.error_function = tf.losses.absolute_difference
        self.session = None
        self.saver = None
        self.use_roi = None
        self.model_to_load = None
        self.batch_size = 32
        self.nodes_graph = dict()
        self.shuffle_buffer = None
        self.continue_session = True
        self.val_size = None
        self.experiment = None
        self.channels_input = None
        self.channels_output = None
        self.test_as_val = None
        self.write_images_tb = False

    def set_experiment(self, experiment):

        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()
        self.channels_input = experiment.channels_input
        self.channels_output = experiment.channels_output
        self.tfrecord_train = os.path.join(experiment.get_records_path(), experiment.get_record_train_name())
        self.continue_session = experiment.get_continue_flag()
        if self.test:
            self.tfrecord_test = os.path.join(experiment.get_records_path(), experiment.get_record_test_name())
        self.logs_path = experiment.get_log_session_path()

        print = self.experiment.open_logger()
        print('##Seting Experiment in trainer##')
        print('-Models paths: %s' % self.models_path)
        print('-Tfrecord Train: %s' % self.tfrecord_train)
        if self.test:
            print('-Tfrecord Test: %s' % self.tfrecord_test)
        else:
            print('-Tfrecord Test: False')
        print('-Logs path: %s' % self.logs_path)
        print('-Continue training: %r' % self.continue_session)

        self.experiment.close_logger()


    @staticmethod
    def gen_summaries_from_dicts(scalars_dict, images_dict):

        def create_scalar_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.scalar(name=name, tensor=tensor))

            return summary

        def create_image_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.image(name=name, tensor=tensor))

            return summary

        summaries = tf.summary.merge(
            create_scalar_summary(scalars_dict) + create_image_summary(images_dict))

        return summaries

    def _create_graph(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        tfrecord_path = tf.placeholder(tf.string, name='tfrecord_path')


        tfrecord_data = get_input_from_record(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                              val_size=self.val_size, buffer_shuffle=self.shuffle_buffer)
        if self.test:
            tfrecord_data_test = get_input_from_record(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                                       buffer_shuffle=self.shuffle_buffer)

        if self.test_as_val:
            tfrecord_data_val = get_input_from_record(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                                       buffer_shuffle=self.shuffle_buffer,repeat=True)

        if self.use_roi:

            if self.val_size:

                inputs_batch, labels_batch, rois_batch, iterator, \
                inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test

            else:
                inputs_batch, labels_batch, rois_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test
                if self.test_as_val:
                    inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data_val

        else:

            if self.val_size:
                inputs_batch, labels_batch, iterator, \
                inputs_batch_val, labels_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
            else:
                inputs_batch, labels_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
                if self.test_as_val:
                    inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data_val

        graph_network = self.network(**self.args_network, input=inputs_batch, is_training=True)
        if self.test:
            graph_network_test = self.network(**self.args_network, input=inputs_batch_test, reuse=True,
                                              is_training=False)

        if self.val_size or self.test_as_val:
            graph_network_val = self.network(**self.args_network, reuse=True, input=inputs_batch_val, is_training=False)
            net_outputs_batch_val = graph_network_val.output

        net_outputs_batch = graph_network.output
        if self.test:
            net_outputs_batch_test = graph_network_test.output

        if self.use_roi:
            if self.val_size or self.test_as_val:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val, rois_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
        else:
            if self.val_size or self.test_as_val:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)
        update_epoch = epoch.assign_add(1)
        self.nodes_graph['global_step'] = global_step
        self.nodes_graph['epoch'] = epoch
        self.nodes_graph['update_epoch'] = update_epoch
        # Node to update epoch

        with tf.control_dependencies(update_ops):
            train_step = self.optimizer(self.learning_rate).minimize(loss_batch, global_step=global_step)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_models_tokeep)

        if self.model_to_load:
            self.saver.restore(self.session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        elif self.continue_session:
            latest_chkpt = tf.train.latest_checkpoint(self.models_path)
            self.saver.restore(self.session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)

        ##SUMMARIES
        dict_scalars_train = {'loss': loss_batch}
        dict_images_train = dict()
        if self.write_images_tb:
            print('writing images in tensorboard')
            for channel_id in range(self.channels_input):
                dict_images_train['input' + str(channel_id)] = tf.expand_dims( inputs_batch[0:3,:,:,channel_id], -1)

            for channel_id in range(self.channels_output):
                dict_images_train['labels' + str(channel_id)] = tf.expand_dims( labels_batch[0:3,:,:,channel_id], -1)
                dict_images_train['outputs' + str(channel_id)] = tf.expand_dims( net_outputs_batch[0:3,:,:,channel_id]  * rois_batch[0:3,:,:,channel_id], -1)




        train_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_train, dict_images_train)

        self.nodes_graph['train_summaries'] = train_summaries_worker

        if self.val_size or self.test_as_val:

            dict_images_val = dict()
            if self.write_images_tb:

                for channel_id in range(self.channels_input):
                    dict_images_val['input_val' + str(channel_id)] = tf.expand_dims( inputs_batch_val[0:3,:,:,channel_id], -1)

                for channel_id in range(self.channels_output):
                    dict_images_val['labels_val' + str(channel_id)] =  tf.expand_dims( labels_batch_val[0:3,:,:,channel_id], -1)
                    dict_images_val['outputs_val' + str(channel_id)] = tf.expand_dims( net_outputs_batch_val[0:3,:,:,channel_id] * rois_batch_val[0:3,:,:,channel_id], -1)



            dict_scalars_val = {'loss_val': loss_batch_val}



            val_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_val, dict_images_val)
            self.nodes_graph['val_summaries'] = val_summaries_worker

        self.nodes_graph['input_batch'] = inputs_batch
        self.nodes_graph['labels_batch'] = labels_batch
        self.nodes_graph['output_batch'] = net_outputs_batch

        self.nodes_graph['iterator'] = iterator

        if self.val_size or self.test_as_val:
            self.nodes_graph['input_batch_val'] = inputs_batch_val
            self.nodes_graph['labels_batch_val'] = labels_batch_val
            self.nodes_graph['iterator_val'] = iterator_val
            self.nodes_graph['output_batch_val'] = net_outputs_batch_val
            self.nodes_graph['loss_batch_val'] = loss_batch_val

        self.nodes_graph['tfrecord_path'] = tfrecord_path
        self.nodes_graph['is_training'] = is_training
        self.nodes_graph['learning_rate'] = learning_rate
        self.nodes_graph['train_step'] = train_step
        self.nodes_graph['loss_batch'] = loss_batch
        if self.test:
            self.nodes_graph['input_batch_test'] = inputs_batch_test
            self.nodes_graph['labels_batch_test'] = labels_batch_test
            self.nodes_graph['iterator_test'] = iterator_test
            self.nodes_graph['output_batch_test'] = net_outputs_batch_test
            self.nodes_graph['loss_batch_test'] = loss_batch_test

    def _log_setup(self):
        print = self.experiment.open_logger()

        print('###Training Setup###')
        print('-Args Network: ')
        print(self.args_network)
        print('-Network details: ')
        print(self.network)
        print('-Optimizer: ')
        print(self.optimizer)
        print('-Error function: ')
        print(self.error_function)
        print('-Use ROI: %r' % self.use_roi)
        print('-Model to load: %s' % self.model_to_load)
        print('-Batch size: %d' % self.batch_size)
        print('-Shuffle Buffer: %s' % self.shuffle_buffer)
        print('-Validation size: %s' % self.val_size)
        print('-Test as validation: %s' % self.test_as_val)
        print('-Max to models to keep: %d' % self.max_to_models_tokeep)

        self.experiment.close_logger()
    def run(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING TRAINING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------

        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
        # summaries_iter = 1

        write_images_tb = True

        while keep_training:



            self.session.run(self.nodes_graph['iterator'].initializer,
                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})



            if self.val_size:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            if self.test_as_val:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})

            print(" epoch number " + str(current_epoch))
            while True:

                try:

                    if self.val_size or self.test_as_val:



                        _, loss, loss_val, train_summaries, val_summaries, step = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['loss_batch_val'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['val_summaries'],
                             self.nodes_graph['global_step']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})


                        #
                        # im,im_val = self.session.run(
                        #     [ self.nodes_graph['input_batch'], self.nodes_graph['labels_batch']],
                        #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})
                        # print(im.max())
                        summary_writer.add_summary(train_summaries, step)
                        summary_writer.add_summary(val_summaries, step)
                    else:
                        _, loss, summaries, step = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['global_step']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})

                        # im = self.session.run(self.nodes_graph['output_batch']
                        #     ,
                        #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                        summary_writer.add_summary(summaries, step)



                except tf.errors.OutOfRangeError:

                    model_dump_path = os.path.join(self.models_path, 'epoch_' + str(current_epoch) + ".ckpt")
                    save_path = self.saver.save(self.session, model_dump_path)
                    print("Model saved in file: %s" % save_path)



                    print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                    print(
                        " -----------------------------------EPOCH FINISHED-----------------------------------------------------------------")
                    if self.test:
                        self.session.run(self.nodes_graph['iterator_test'].initializer,
                                         feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})


                        print(" epoch testing " + str(current_epoch))

                        loss_val_list = []

                        while True:

                            try:

                                loss_test = self.session.run(
                                    self.nodes_graph['loss_batch_test'])

                                loss_val_list.append(loss_test)



                            except tf.errors.OutOfRangeError:

                                test_loss_mean = np.mean(loss_val_list)
                                summary = tf.Summary()
                                summary.value.add(tag="avg_test_loss_epoch", simple_value=test_loss_mean)
                                summary_writer.add_summary(summary, current_epoch)
                                print('Test Loss: %f' % test_loss_mean)

                                print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                                print(
                                    " -----------------------------------EPOCH testing FINISHED-----------------------------------------------------------------")

                                break

                    # current_epoch += 1
                    _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
                    if self.epochs is not None:
                        if current_epoch > self.epochs:
                            keep_training = False

                    break

        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()

    def run_debug_record(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING RECORD DEBUGING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------

        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
        # summaries_iter = 1

        while keep_training:
            self.session.run(self.nodes_graph['iterator'].initializer,
                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            if self.val_size:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            print(" epoch number " + str(current_epoch))
            import cv2
            while True:

                try:

                    if self.val_size:


                        im_in,im_in_val,im_out,im_out_val = self.session.run(
                            [ self.nodes_graph['input_batch'],self.nodes_graph['input_batch_val'], self.nodes_graph['output_batch'], self.nodes_graph['output_batch_val']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                    else:

                        im_in, im_out = self.session.run([self.nodes_graph['input_batch'],self.nodes_graph['output_batch']]
                            ,
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                    im_imshow = im_in[0]
                    im_out_imshow = im_out[0]
                    cv2.imshow('input 1', im_imshow[:,:,0])
                    cv2.imshow('input 2', im_imshow[:,:,1])
                    cv2.imshow('input 3', im_imshow[:,:,2])
                    cv2.imshow('input 4', im_imshow[:,:,3])
                    cv2.imshow('input 5', im_imshow[:,:,4])
                    cv2.waitKey(1000)
                    print("debugeamos aqui")


                except tf.errors.OutOfRangeError:


                    break

        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()


class RegressionTrainer_patch3d:
    def __init__(self):

        self.epochs = None
        self.network = None
        self.args_network = None
        self.test = False
        self.learning_rate = None
        self.models_path = os.path.join(os.getcwd(), 'models')
        self.tfrecord_train = os.path.join(os.getcwd(), 'records', 'train.tfrecord')
        self.tfrecord_test = os.path.join(os.getcwd(), 'records', 'test.tfrecord')
        self.logs_path = os.path.join(os.getcwd(), 'logs_tb', 'test.tfrecord')
        self.max_to_models_tokeep = 25
        self.optimizer = tf.train.AdamOptimizer
        self.error_function = tf.losses.absolute_difference
        self.session = None
        self.saver = None
        self.use_roi = None
        self.model_to_load = None
        self.batch_size = 32
        self.nodes_graph = dict()
        self.shuffle_buffer = None
        self.continue_session = True
        self.val_size = None
        self.experiment = None
        self.channels_input = None
        self.channels_output = None
        self.test_as_val = None
        self.write_images_tb = False
        self.step_test = 500
        self.training_rounds  = 20
        self.train_per_epoch = True
        self.repeat_train_data = False

    def set_experiment(self, experiment):

        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()
        self.channels_input = experiment.channels_input
        self.channels_output = experiment.channels_output
        self.tfrecord_train = os.path.join(experiment.get_records_path(), experiment.get_record_train_name())
        self.continue_session = experiment.get_continue_flag()
        if self.test:
            self.tfrecord_test = os.path.join(experiment.get_records_path(), experiment.get_record_test_name())
        self.logs_path = experiment.get_log_session_path()

        print = self.experiment.open_logger()
        print('##Seting Experiment in trainer##')
        print('-Models paths: %s' % self.models_path)
        print('-Tfrecord Train: %s' % self.tfrecord_train)
        if self.test:
            print('-Tfrecord Test: %s' % self.tfrecord_test)
        else:
            print('-Tfrecord Test: False')
        print('-Logs path: %s' % self.logs_path)
        print('-Continue training: %r' % self.continue_session)

        self.experiment.close_logger()


    @staticmethod
    def gen_summaries_from_dicts(scalars_dict, images_dict):

        def create_scalar_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.scalar(name=name, tensor=tensor))

            return summary

        def create_image_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.image(name=name, tensor=tensor))

            return summary

        summaries = tf.summary.merge(
            create_scalar_summary(scalars_dict) + create_image_summary(images_dict))

        return summaries

    def _create_graph(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        tfrecord_path = tf.placeholder(tf.string, name='tfrecord_path')


        tfrecord_data = get_input_from_record_patch3d(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                              val_size=self.val_size, buffer_shuffle=self.shuffle_buffer,repeat = self.repeat_train_data)
        if self.test:
            tfrecord_data_test = get_input_from_record_patch3d(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                                       buffer_shuffle=self.shuffle_buffer)

        if self.test_as_val:
            tfrecord_data_val = get_input_from_record_patch3d(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                                       buffer_shuffle=self.shuffle_buffer,repeat=True)

        if self.use_roi:

            if self.val_size:

                inputs_batch, labels_batch, rois_batch, iterator, \
                inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test

            else:
                inputs_batch, labels_batch, rois_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test
                if self.test_as_val:
                    inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data_val

        else:

            if self.val_size:
                inputs_batch, labels_batch, iterator, \
                inputs_batch_val, labels_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
            else:
                inputs_batch, labels_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
                if self.test_as_val:
                    inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data_val

        graph_network = self.network(**self.args_network, input=inputs_batch, is_training=True)
        if self.test:
            graph_network_test = self.network(**self.args_network, input=inputs_batch_test, reuse=True,
                                              is_training=False)

        if self.val_size or self.test_as_val:
            graph_network_val = self.network(**self.args_network, reuse=True, input=inputs_batch_val, is_training=False)
            net_outputs_batch_val = graph_network_val.output

        net_outputs_batch = graph_network.output
        if self.test:
            net_outputs_batch_test = graph_network_test.output

        if self.use_roi:
            if self.val_size or self.test_as_val:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val, rois_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
        else:
            if self.val_size or self.test_as_val:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)
        update_epoch = epoch.assign_add(1)
        self.nodes_graph['global_step'] = global_step
        self.nodes_graph['epoch'] = epoch
        self.nodes_graph['update_epoch'] = update_epoch
        # Node to update epoch

        with tf.control_dependencies(update_ops):
            train_step = self.optimizer(self.learning_rate).minimize(loss_batch, global_step=global_step)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_models_tokeep)

        if self.model_to_load:
            self.saver.restore(self.session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        elif self.continue_session:
            latest_chkpt = tf.train.latest_checkpoint(self.models_path)
            self.saver.restore(self.session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)

        ##SUMMARIES
        dict_scalars_train = {'loss': loss_batch}
        dict_images_train = dict()
        if self.write_images_tb:
            print('writing images in tensorboard')
            for channel_id in range(self.channels_input):
                dict_images_train['input' + str(channel_id)] = tf.expand_dims(inputs_batch[0:3,:,:,16,channel_id],-1)

            for channel_id in range(self.channels_output):
                dict_images_train['labels' + str(channel_id)] =  tf.expand_dims(labels_batch[0:3,:,:,16,channel_id],-1)
                dict_images_train['outputs' + str(channel_id)] =tf.expand_dims(net_outputs_batch[0:3,:,:,16,channel_id]  * rois_batch[0:3,:,:,16,channel_id],-1)




        train_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_train, dict_images_train)

        self.nodes_graph['train_summaries'] = train_summaries_worker

        if self.val_size or self.test_as_val:

            dict_images_val = dict()
            if self.write_images_tb:

                for channel_id in range(self.channels_input):
                    dict_images_val['input_val' + str(channel_id)] =tf.expand_dims( inputs_batch_val[0:3,:,:,16,channel_id],-1)

                for channel_id in range(self.channels_output):
                    dict_images_val['labels_val' + str(channel_id)] =  tf.expand_dims(labels_batch_val[0:3,:,:,16,channel_id],-1)
                    dict_images_val['outputs_val' + str(channel_id)] = tf.expand_dims(net_outputs_batch_val[0:3,:,:,16,channel_id] * rois_batch_val[0:3,:,:,16,channel_id],-1)



            dict_scalars_val = {'loss_val': loss_batch_val}



            val_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_val, dict_images_val)
            self.nodes_graph['val_summaries'] = val_summaries_worker

        self.nodes_graph['input_batch'] = inputs_batch
        self.nodes_graph['labels_batch'] = labels_batch
        self.nodes_graph['output_batch'] = net_outputs_batch

        self.nodes_graph['iterator'] = iterator

        if self.val_size or self.test_as_val:
            self.nodes_graph['input_batch_val'] = inputs_batch_val
            self.nodes_graph['labels_batch_val'] = labels_batch_val
            self.nodes_graph['iterator_val'] = iterator_val
            self.nodes_graph['output_batch_val'] = net_outputs_batch_val
            self.nodes_graph['loss_batch_val'] = loss_batch_val

        self.nodes_graph['tfrecord_path'] = tfrecord_path
        self.nodes_graph['is_training'] = is_training
        self.nodes_graph['learning_rate'] = learning_rate
        self.nodes_graph['train_step'] = train_step
        self.nodes_graph['loss_batch'] = loss_batch
        if self.test:
            self.nodes_graph['input_batch_test'] = inputs_batch_test
            self.nodes_graph['labels_batch_test'] = labels_batch_test
            self.nodes_graph['iterator_test'] = iterator_test
            self.nodes_graph['output_batch_test'] = net_outputs_batch_test
            self.nodes_graph['loss_batch_test'] = loss_batch_test

    def _log_setup(self):
        print = self.experiment.open_logger()

        print('###Training Setup###')
        print('-Args Network: ')
        print(self.args_network)
        print('-Network details: ')
        print(self.network)
        print('-Optimizer: ')
        print(self.optimizer)
        print('-Error function: ')
        print(self.error_function)
        print('-Use ROI: %r' % self.use_roi)
        print('-Model to load: %s' % self.model_to_load)
        print('-Batch size: %d' % self.batch_size)
        print('-Shuffle Buffer: %s' % self.shuffle_buffer)
        print('-Validation size: %s' % self.val_size)
        print('-Test as validation: %s' % self.test_as_val)
        print('-Max to models to keep: %d' % self.max_to_models_tokeep)

        self.experiment.close_logger()
    def run(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING TRAINING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------
        if self.train_per_epoch:


            # TRAIN LOOP------------------------------------------------------

            keep_training = True

            _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
            # summaries_iter = 1

            write_images_tb = True

            while keep_training:



                self.session.run(self.nodes_graph['iterator'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})



                if self.val_size:
                    self.session.run(self.nodes_graph['iterator_val'].initializer,
                                     feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

                if self.test_as_val:
                    self.session.run(self.nodes_graph['iterator_val'].initializer,
                                     feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})

                print(" epoch number " + str(current_epoch))
                while True:

                    try:

                        if self.val_size or self.test_as_val:



                            _, loss, loss_val, train_summaries, val_summaries, step = self.session.run(
                                [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                                 self.nodes_graph['loss_batch_val'],
                                 self.nodes_graph['train_summaries'], self.nodes_graph['val_summaries'],
                                 self.nodes_graph['global_step']],
                                feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                           self.nodes_graph['is_training']: True})


                            #
                            # im,im_val = self.session.run(
                            #     [ self.nodes_graph['input_batch'], self.nodes_graph['labels_batch']],
                            #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})
                            # print(im.max())
                            summary_writer.add_summary(train_summaries, step)
                            summary_writer.add_summary(val_summaries, step)
                        else:
                            _, loss, summaries, step = self.session.run(
                                [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                                 self.nodes_graph['train_summaries'], self.nodes_graph['global_step']],
                                feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                           self.nodes_graph['is_training']: True})

                            # im = self.session.run(self.nodes_graph['output_batch']
                            #     ,
                            #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                            summary_writer.add_summary(summaries, step)



                    except tf.errors.OutOfRangeError:

                        model_dump_path = os.path.join(self.models_path, 'epoch_' + str(current_epoch) + ".ckpt")
                        save_path = self.saver.save(self.session, model_dump_path)
                        print("Model saved in file: %s" % save_path)



                        print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                        print(
                            " -----------------------------------EPOCH FINISHED-----------------------------------------------------------------")
                        if self.test:
                            self.session.run(self.nodes_graph['iterator_test'].initializer,
                                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})


                            print(" epoch testing " + str(current_epoch))

                            loss_val_list = []

                            while True:

                                try:

                                    loss_test = self.session.run(
                                        self.nodes_graph['loss_batch_test'])

                                    loss_val_list.append(loss_test)



                                except tf.errors.OutOfRangeError:

                                    test_loss_mean = np.mean(loss_val_list)
                                    summary = tf.Summary()
                                    summary.value.add(tag="avg_test_loss_epoch", simple_value=test_loss_mean)
                                    summary_writer.add_summary(summary, current_epoch)
                                    print('Test Loss: %f' % test_loss_mean)

                                    print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                                    print(
                                        " -----------------------------------EPOCH testing FINISHED-----------------------------------------------------------------")

                                    break

                        # current_epoch += 1
                        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
                        if self.epochs is not None:
                            if current_epoch > self.epochs:
                                keep_training = False

                        break


        else:

            # TRAIN LOOP------------------------------------------------------

            keep_training = True

            _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
            # summaries_iter = 1


            current_train_round = 0
            self.session.run(self.nodes_graph['iterator'].initializer,
                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})
            if self.val_size:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            if self.test_as_val:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})
            while keep_training:





                keep_train_round = True

                current_train_round += 1

                print(" train round number " + str(current_train_round))
                while keep_train_round:


                        if self.val_size or self.test_as_val:

                            _, loss, loss_val, train_summaries, val_summaries, step = self.session.run(
                                [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                                 self.nodes_graph['loss_batch_val'],
                                 self.nodes_graph['train_summaries'], self.nodes_graph['val_summaries'],
                                 self.nodes_graph['global_step']],
                                feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                           self.nodes_graph['is_training']: True})

                            #
                            # im,im_val = self.session.run(
                            #     [ self.nodes_graph['input_batch'], self.nodes_graph['labels_batch']],
                            #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})
                            # print(im.max())
                            summary_writer.add_summary(train_summaries, step)
                            summary_writer.add_summary(val_summaries, step)
                        else:
                            _, loss, summaries, step = self.session.run(
                                [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                                 self.nodes_graph['train_summaries'], self.nodes_graph['global_step']],
                                feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                           self.nodes_graph['is_training']: True})

                            # im = self.session.run(self.nodes_graph['output_batch']
                            #     ,
                            #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                            summary_writer.add_summary(summaries, step)

                        if (step+1)%self.step_test==0:



                            model_dump_path = os.path.join(self.models_path, 'epoch_' + str(step) + ".ckpt")
                            save_path = self.saver.save(self.session, model_dump_path)
                            print("Model saved in file: %s" % save_path)

                            print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                            print(
                                " -----------------------------------TRAINING ROUND FINISHED-----------------------------------------------------------------")
                            if self.test:
                                self.session.run(self.nodes_graph['iterator_test'].initializer,
                                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})

                                print(" epoch testing " + str(current_train_round))

                                loss_val_list = []

                                while True:

                                    try:

                                        loss_test = self.session.run(
                                            self.nodes_graph['loss_batch_test'])

                                        loss_val_list.append(loss_test)



                                    except tf.errors.OutOfRangeError:

                                        test_loss_mean = np.mean(loss_val_list)
                                        summary = tf.Summary()
                                        summary.value.add(tag="avg_test_loss_epoch", simple_value=test_loss_mean)
                                        summary_writer.add_summary(summary, current_train_round)
                                        print('Test Loss: %f' % test_loss_mean)

                                        print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                                        print(
                                            " -----------------------------------EPOCH testing FINISHED-----------------------------------------------------------------")

                                        break



                            if current_train_round == self.training_rounds:
                                keep_training = False

                            break



        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()

    def run_debug_record(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING RECORD DEBUGING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------

        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
        # summaries_iter = 1

        while keep_training:
            self.session.run(self.nodes_graph['iterator'].initializer,
                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            if self.val_size:
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            print(" epoch number " + str(current_epoch))
            import cv2
            while True:

                try:

                    if self.val_size:


                        im_in,im_in_val,im_out,im_out_val = self.session.run(
                            [ self.nodes_graph['input_batch'],self.nodes_graph['input_batch_val'], self.nodes_graph['output_batch'], self.nodes_graph['output_batch_val']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                    else:

                        im_in, im_out = self.session.run([self.nodes_graph['input_batch'],self.nodes_graph['output_batch']]
                            ,
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                    im_imshow = im_in[0]
                    im_out_imshow = im_out[0]
                    cv2.imshow('input 1', im_imshow[:,:,0])
                    cv2.imshow('input 2', im_imshow[:,:,1])
                    cv2.imshow('input 3', im_imshow[:,:,2])
                    cv2.imshow('input 4', im_imshow[:,:,3])
                    cv2.imshow('input 5', im_imshow[:,:,4])
                    cv2.waitKey(1000)
                    print("debugeamos aqui")


                except tf.errors.OutOfRangeError:


                    break

        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()


class RegressionTrainer_patch3d_dataset:
    def __init__(self):

        self.epochs = None
        self.network = None
        self.args_network = None
        self.test = False
        self.learning_rate = None
        self.models_path = os.path.join(os.getcwd(), 'models')
        self.logs_path = os.path.join(os.getcwd(), 'logs_tb', 'test.tfrecord')
        self.max_to_models_tokeep = 5
        self.optimizer = tf.train.AdamOptimizer
        self.error_function = tf.losses.absolute_difference
        self.session = None
        self.saver = None
        self.use_roi = None
        self.model_to_load = None
        self.batch_size = 32
        self.nodes_graph = dict()
        self.shuffle_buffer = None
        self.continue_session = True
        self.val_size = None
        self.experiment = None
        self.channels_input = None
        self.channels_output = None
        self.test_as_val = None
        self.write_images_tb = False
        self.train_dataset = None
        self.test_dataset = None
        self.steps_per_epoch = None
        self.steps_per_epoch_test = None
        self.max_epochs = None


    def set_experiment(self, experiment):

        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()
        self.channels_input = experiment.channels_input
        self.channels_output = experiment.channels_output
        self.continue_session = experiment.get_continue_flag()

        self.logs_path = experiment.get_log_session_path()

        print = self.experiment.open_logger()
        print('##Seting Experiment in trainer##')
        print('-Models paths: %s' % self.models_path)
        print('-Logs path: %s' % self.logs_path)
        print('-Continue training: %r' % self.continue_session)

        self.experiment.close_logger()


    @staticmethod
    def gen_summaries_from_dicts(scalars_dict, images_dict):

        def create_scalar_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.scalar(name=name, tensor=tensor))

            return summary

        def create_image_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.image(name=name, tensor=tensor))

            return summary

        summaries = tf.summary.merge(
            create_scalar_summary(scalars_dict) + create_image_summary(images_dict))

        return summaries

    def _create_graph(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        tfrecord_data = get_input_from_dataset(self.train_dataset)


        if self.test:
            tfrecord_data_test = get_input_from_dataset(self.test_dataset)

        if self.test_as_val:
            tfrecord_data_val = get_input_from_dataset(self.test_dataset)

        if self.use_roi:

            if self.val_size:

                inputs_batch, labels_batch, rois_batch, iterator, \
                inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test

            else:
                inputs_batch, labels_batch, rois_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, rois_batch_test, iterator_test = tfrecord_data_test
                if self.test_as_val:
                    inputs_batch_val, labels_batch_val, rois_batch_val, iterator_val = tfrecord_data_val

        else:

            if self.val_size:
                inputs_batch, labels_batch, iterator, \
                inputs_batch_val, labels_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
            else:
                inputs_batch, labels_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, iterator_test = tfrecord_data_test
                if self.test_as_val:
                    inputs_batch_val, labels_batch_val, iterator_val = tfrecord_data_val

        graph_network = self.network(**self.args_network, input=inputs_batch, is_training=True)
        if self.test:
            graph_network_test = self.network(**self.args_network, input=inputs_batch_test, reuse=True,
                                              is_training=False)

        if self.val_size or self.test_as_val:
            graph_network_val = self.network(**self.args_network, reuse=True, input=inputs_batch_val, is_training=False)
            net_outputs_batch_val = graph_network_val.output

        net_outputs_batch = graph_network.output
        if self.test:
            net_outputs_batch_test = graph_network_test.output

        if self.use_roi:
            if self.val_size or self.test_as_val:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val, rois_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
        else:
            if self.val_size or self.test_as_val:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)
        update_epoch = epoch.assign_add(1)
        self.nodes_graph['global_step'] = global_step
        self.nodes_graph['epoch'] = epoch
        self.nodes_graph['update_epoch'] = update_epoch
        # Node to update epoch

        with tf.control_dependencies(update_ops):
            train_step = self.optimizer(self.learning_rate).minimize(loss_batch, global_step=global_step)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_models_tokeep)

        if self.model_to_load:
            self.saver.restore(self.session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        elif self.continue_session:
            latest_chkpt = tf.train.latest_checkpoint(self.models_path)
            self.saver.restore(self.session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)

        ##SUMMARIES
        dict_scalars_train = {'loss': loss_batch}
        dict_images_train = dict()
        if self.write_images_tb:
            print('writing images in tensorboard')
            for channel_id in range(self.channels_input):
                dict_images_train['input' + str(channel_id)] = tf.expand_dims(inputs_batch[0:3,:,:,16,channel_id],-1)

            for channel_id in range(self.channels_output):
                dict_images_train['labels' + str(channel_id)] =  tf.expand_dims(labels_batch[0:3,:,:,16,channel_id],-1)
                if self.use_roi:
                    dict_images_train['outputs' + str(channel_id)] =tf.expand_dims(net_outputs_batch[0:3,:,:,16,channel_id]  * rois_batch[0:3,:,:,16,channel_id],-1)
                else:
                    dict_images_train['outputs' + str(channel_id)] = tf.expand_dims(
                        net_outputs_batch[0:3, :, :, 16, channel_id], -1)



        train_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_train, dict_images_train)

        self.nodes_graph['train_summaries'] = train_summaries_worker

        if self.val_size or self.test_as_val:

            dict_images_val = dict()
            if self.write_images_tb:

                for channel_id in range(self.channels_input):
                    dict_images_val['input_val' + str(channel_id)] =tf.expand_dims( inputs_batch_val[0:3,:,:,16,channel_id],-1)

                for channel_id in range(self.channels_output):
                    dict_images_val['labels_val' + str(channel_id)] =  tf.expand_dims(labels_batch_val[0:3,:,:,16,channel_id],-1)
                    if self.use_roi:
                        dict_images_val['outputs_val' + str(channel_id)] = tf.expand_dims(net_outputs_batch_val[0:3,:,:,16,channel_id] * rois_batch_val[0:3,:,:,16,channel_id],-1)
                    else:
                        dict_images_val['outputs_val' + str(channel_id)] = tf.expand_dims(
                            net_outputs_batch_val[0:3, :, :, 16, channel_id], -1)


            dict_scalars_val = {'loss_val': loss_batch_val}

            self.nodes_graph['net_output_batch_val'] = labels_batch_val

            val_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_val, dict_images_val)
            self.nodes_graph['val_summaries'] = val_summaries_worker

        self.nodes_graph['input_batch'] = inputs_batch
        self.nodes_graph['labels_batch'] = labels_batch
        self.nodes_graph['output_batch'] = net_outputs_batch

        self.nodes_graph['iterator'] = iterator

        if self.val_size or self.test_as_val:
            self.nodes_graph['input_batch_val'] = inputs_batch_val
            self.nodes_graph['labels_batch_val'] = labels_batch_val
            self.nodes_graph['iterator_val'] = iterator_val
            self.nodes_graph['output_batch_val'] = net_outputs_batch_val
            self.nodes_graph['loss_batch_val'] = loss_batch_val


        self.nodes_graph['is_training'] = is_training
        self.nodes_graph['learning_rate'] = learning_rate
        self.nodes_graph['train_step'] = train_step
        self.nodes_graph['loss_batch'] = loss_batch
        if self.test:
            self.nodes_graph['input_batch_test'] = inputs_batch_test
            self.nodes_graph['labels_batch_test'] = labels_batch_test
            self.nodes_graph['iterator_test'] = iterator_test
            self.nodes_graph['output_batch_test'] = net_outputs_batch_test
            self.nodes_graph['loss_batch_test'] = loss_batch_test

    def _log_setup(self):
        print = self.experiment.open_logger()

        print('###Training Setup###')
        print('-Args Network: ')
        print(self.args_network)
        print('-Network details: ')
        print(self.network)
        print('-Optimizer: ')
        print(self.optimizer)
        print('-Error function: ')
        print(self.error_function)
        print('-Use ROI: %r' % self.use_roi)
        print('-Model to load: %s' % self.model_to_load)
        print('-Batch size: %d' % self.batch_size)
        print('-Shuffle Buffer: %s' % self.shuffle_buffer)
        print('-Validation size: %s' % self.val_size)
        print('-Test as validation: %s' % self.test_as_val)
        print('-Max to models to keep: %d' % self.max_to_models_tokeep)

        self.experiment.close_logger()
    def run(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING TRAINING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------



        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        epoch = 0



        self.session.run(self.nodes_graph['iterator'].initializer)
        if self.val_size:
            self.session.run(self.nodes_graph['iterator_val'].initializer)

        if self.test_as_val:
            self.session.run(self.nodes_graph['iterator_val'].initializer)
        while keep_training:

            epoch += 1



            keep_train_round = True



            print(" epoch " + str(epoch))
            while keep_train_round:


                    if self.val_size or self.test_as_val:





                        _, loss, loss_val, train_summaries, val_summaries, step = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['loss_batch_val'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['val_summaries'],
                             self.nodes_graph['global_step']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})

                        #
                        # im,im_val = self.session.run(
                        #     [ self.nodes_graph['input_batch'], self.nodes_graph['labels_batch']],
                        #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})
                        # print(im.max())
                        summary_writer.add_summary(train_summaries, step)
                        summary_writer.add_summary(val_summaries, step)
                    else:
                        _, loss, summaries, step = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['global_step']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})

                        # im = self.session.run(self.nodes_graph['output_batch']
                        #     ,
                        #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                        summary_writer.add_summary(summaries, step)

                    if (step+1)%self.steps_per_epoch==0:



                        model_dump_path = os.path.join(self.models_path, 'epoch_' + str(epoch) + ".ckpt")
                        save_path = self.saver.save(self.session, model_dump_path)
                        print("Model saved in file: %s" % save_path)

                        print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                        print(
                            " -----------------------------------TRAINING ROUND FINISHED-----------------------------------------------------------------")
                        if self.test:
                            self.session.run(self.nodes_graph['iterator_test'].initializer)

                            print(" epoch testing " + str(epoch))

                            loss_val_list = []

                            while True:



                                    loss_test = self.session.run(
                                        self.nodes_graph['loss_batch_test'])

                                    loss_val_list.append(loss_test)



                                    if (step+1)%self.steps_per_epoch_test==0:

                                        test_loss_mean = np.mean(loss_val_list)
                                        summary = tf.Summary()
                                        summary.value.add(tag="avg_test_loss_epoch", simple_value=test_loss_mean)
                                        summary_writer.add_summary(summary, epoch)
                                        print('Test Loss: %f' % test_loss_mean)

                                        print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                                        print(
                                            " -----------------------------------EPOCH testing FINISHED-----------------------------------------------------------------")

                                        break



                        if self.max_epochs == epoch:
                            keep_training = False

                        break



        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()

    def run_debug_record(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING RECORD DEBUGING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

  # ---------------------------------------------------------------------

        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])

        while keep_training:
            self.session.run(self.nodes_graph['iterator'].initializer)

            if self.val_size or self.test_as_val:
                self.session.run(self.nodes_graph['iterator_val'].initializer)

            print(" epoch number " + str(current_epoch))
            import cv2
            counter_iters = 0
            while True:
                counter_iters+=1
                try:
                    print('counter iters:')
                    print(counter_iters)
                    if self.val_size or self.test_as_val:


                        im_in,im_in_val,im_out,im_out_val, im_label, im_label_val = self.session.run(
                            [ self.nodes_graph['input_batch'],self.nodes_graph['input_batch_val'], self.nodes_graph['output_batch'], self.nodes_graph['output_batch_val']
                              , self.nodes_graph['labels_batch'], self.nodes_graph['labels_batch_val']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                    else:

                        im_in, im_out = self.session.run([self.nodes_graph['input_batch'],self.nodes_graph['output_batch']]
                            ,
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})




                    # im_imshow = im_in[0]
                    # im_out_imshow = im_out[0]
                    # cv2.imshow('input 1', im_imshow[:,:,0])
                    # cv2.imshow('input 2', im_imshow[:,:,1])
                    # cv2.imshow('input 3', im_imshow[:,:,2])
                    # cv2.imshow('input 4', im_imshow[:,:,3])
                    # cv2.imshow('input 5', im_imshow[:,:,4])
                    # cv2.waitKey(1000)
                    # print("debugeamos aqui")


                except tf.errors.OutOfRangeError:
                    print('Final del epoch')
                    print(counter_iters)

                    break

        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()





class RegressionTrainer3D2D:
    def __init__(self):

        self.epochs = None
        self.network = None
        self.args_network = None
        self.test = False
        self.learning_rate = None
        self.models_path = os.path.join(os.getcwd(), 'models')
        self.tfrecord_train = os.path.join(os.getcwd(), 'records', 'train3d.tfrecord')
        self.tfrecord_test = os.path.join(os.getcwd(), 'records', 'test3d.tfrecord')
        self.logs_path = os.path.join(os.getcwd(), 'logs_tb', 'test.tfrecord')
        self.max_to_models_tokeep = 10
        self.optimizer = tf.train.AdamOptimizer
        self.error_function = tf.losses.absolute_difference
        self.session = None
        self.saver = None
        self.use_roi = None
        self.model_to_load = None
        self.batch_size = 32
        self.nodes_graph = dict()
        self.shuffle_buffer = None
        self.continue_session = True
        self.val_size = None
        self.experiment = None
        self.channels_input = None
        self.channels_output = None

    def set_experiment(self, experiment):

        assert isinstance(experiment, Experiment)
        self.experiment = experiment
        self.models_path = experiment.get_models_session_path()
        self.channels_input = experiment.channels_input
        self.channels_output = experiment.channels_output
        self.tfrecord_train = os.path.join(experiment.get_records_path(), experiment.get_record_train_name())
        self.continue_session = experiment.get_continue_flag()
        if self.test:
            self.tfrecord_test = os.path.join(experiment.get_records_path(), experiment.get_record_test_name())
        self.logs_path = experiment.get_log_session_path()

        print = self.experiment.open_logger()
        print('##Seting Experiment in trainer##')
        print('-Models paths: %s' % self.models_path)
        print('-Tfrecord Train: %s' % self.tfrecord_train)
        if self.test:
            print('-Tfrecord Test: %s' % self.tfrecord_test)
        else:
            print('-Tfrecord Test: False')
        print('-Logs path: %s' % self.logs_path)
        print('-Continue training: %r' % self.continue_session)

        self.experiment.close_logger()


    @staticmethod
    def gen_summaries_from_dicts(scalars_dict, images_dict):

        def create_scalar_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.scalar(name=name, tensor=tensor))

            return summary

        def create_image_summary(dict):

            summary = []
            for name, tensor in zip(dict.keys(), dict.values()):
                summary.append(tf.summary.image(name=name, tensor=tensor))

            return summary

        summaries = tf.summary.merge(
            create_scalar_summary(scalars_dict) + create_image_summary(images_dict))

        return summaries

    def _create_graph(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        tfrecord_path = tf.placeholder(tf.string, name='tfrecord_path')

        tfrecord_data = get_input_from_record3d2d(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                              val_size=self.val_size, buffer_shuffle=self.shuffle_buffer)
        if self.test:
            tfrecord_data_test = get_input_from_record3d2d(tfrecord_path, self.batch_size, read_roi=self.use_roi,
                                                       buffer_shuffle=self.shuffle_buffer)

        if self.use_roi:

            if self.val_size:

                inputs_batch, labels_batch, vols_batch, slice_id_batch,rois_batch, iterator, \
                inputs_batch_val, labels_batch_val, vols_batch_val,slice_id_batch_val, rois_batch_val, iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test,vols_batch_test, slice_id_batch_test,rois_batch_test, iterator_test = tfrecord_data_test

            else:
                inputs_batch, labels_batch,vols_batch, slice_id_batch,rois_batch, iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, vols_batch_test, slice_id_batch_test,rois_batch_test, iterator_test = tfrecord_data_test
        else:

            if self.val_size:
                inputs_batch, labels_batch, vols_batch,slice_id_batch,iterator, \
                inputs_batch_val, labels_batch_val, vols_batch_val, slice_id_batch_val,iterator_val = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test,vols_batch_test, slice_id_batch_test,iterator_test = tfrecord_data_test
            else:
                inputs_batch, labels_batch, vols_batch,slice_id_batch,iterator = tfrecord_data
                if self.test:
                    inputs_batch_test, labels_batch_test, vols_batch_test,slice_id_batch_test,iterator_test = tfrecord_data_test

        graph_network = self.network(**self.args_network, vol_input = vols_batch, imgs_input = inputs_batch,slice_id = slice_id_batch , is_training=True)
        if self.test:
            graph_network_test = self.network(**self.args_network, vol_input = vols_batch_test, imgs_input = inputs_batch_test,slice_id = slice_id_batch_test, reuse=True,
                                              is_training=False)

        if self.val_size:
            graph_network_val = self.network(**self.args_network, reuse=True,vol_input = vols_batch_val, imgs_input = inputs_batch_val,slice_id = slice_id_batch_val, is_training=False)
            net_outputs_batch_val = graph_network_val.output

        net_outputs_batch = graph_network.output
        if self.test:
            net_outputs_batch_test = graph_network_test.output

        if self.use_roi:
            if self.val_size:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val, rois_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch, rois_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test, rois_batch_test)
        else:
            if self.val_size:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                loss_batch_val = self.error_function(net_outputs_batch_val,
                                                     labels_batch_val)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

            else:
                loss_batch = self.error_function(net_outputs_batch,
                                                 labels_batch)
                if self.test:
                    loss_batch_test = self.error_function(net_outputs_batch_test,
                                                          labels_batch_test)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)
        update_epoch = epoch.assign_add(1)
        self.nodes_graph['global_step'] = global_step
        self.nodes_graph['epoch'] = epoch
        self.nodes_graph['update_epoch'] = update_epoch
        # Node to update epoch

        with tf.control_dependencies(update_ops):
            train_step = self.optimizer(self.learning_rate).minimize(loss_batch, global_step=global_step)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_models_tokeep)

        if self.model_to_load:
            self.saver.restore(self.session, self.model_to_load)
            print("Model %s restored." % self.model_to_load)
        elif self.continue_session:
            latest_chkpt = tf.train.latest_checkpoint(self.models_path)
            self.saver.restore(self.session, latest_chkpt)
            print("Model %s restored." % latest_chkpt)

        ##SUMMARIES
        dict_scalars_train = {'loss': loss_batch}
        dict_images_train = dict()

        for channel_id in range(self.channels_input):
            dict_images_train['input' + str(channel_id)] = tf.expand_dims( inputs_batch[0:3,:,:,channel_id], -1)

        for channel_id in range(self.channels_output):
            dict_images_train['labels' + str(channel_id)] = tf.expand_dims( labels_batch[0:3,:,:,channel_id], -1)
            dict_images_train['outputs' + str(channel_id)] = tf.expand_dims( net_outputs_batch[0:3,:,:,channel_id]  * rois_batch[0:3,:,:,channel_id], -1)




        train_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_train, dict_images_train)

        self.nodes_graph['train_summaries'] = train_summaries_worker

        if self.val_size:

            dict_images_val = dict()


            for channel_id in range(self.channels_input):
                dict_images_val['input_val' + str(channel_id)] = tf.expand_dims( inputs_batch_val[0:3,:,:,channel_id], -1)

            for channel_id in range(self.channels_output):
                dict_images_val['labels_val' + str(channel_id)] =  tf.expand_dims( labels_batch_val[0:3,:,:,channel_id], -1)
                dict_images_val['outputs_val' + str(channel_id)] = tf.expand_dims( net_outputs_batch_val[0:3,:,:,channel_id] * rois_batch_val[0:3,:,:,channel_id], -1)



            dict_scalars_val = {'loss_val': loss_batch_val}



            val_summaries_worker = self.gen_summaries_from_dicts(dict_scalars_val, dict_images_val)
            self.nodes_graph['val_summaries'] = val_summaries_worker

        self.nodes_graph['input_batch'] = inputs_batch
        self.nodes_graph['labels_batch'] = labels_batch
        self.nodes_graph['output_batch'] = net_outputs_batch

        self.nodes_graph['iterator'] = iterator
        if self.val_size:
            self.nodes_graph['input_batch_val'] = inputs_batch_val
            self.nodes_graph['labels_batch_val'] = labels_batch_val
            self.nodes_graph['iterator_val'] = iterator_val
            self.nodes_graph['output_batch_val'] = net_outputs_batch_val
            self.nodes_graph['loss_batch_val'] = loss_batch_val

        self.nodes_graph['tfrecord_path'] = tfrecord_path
        self.nodes_graph['is_training'] = is_training
        self.nodes_graph['learning_rate'] = learning_rate
        self.nodes_graph['train_step'] = train_step
        self.nodes_graph['loss_batch'] = loss_batch
        if self.test:
            self.nodes_graph['input_batch_test'] = inputs_batch_test
            self.nodes_graph['labels_batch_test'] = labels_batch_test
            self.nodes_graph['iterator_test'] = iterator_test
            self.nodes_graph['output_batch_test'] = net_outputs_batch_test
            self.nodes_graph['loss_batch_test'] = loss_batch_test

    def _log_setup(self):
        print = self.experiment.open_logger()

        print('###Training Setup###')
        print('-Args Network: ')
        print(self.args_network)
        print('-Network details: ')
        print(self.network)
        print('-Optimizer: ')
        print(self.optimizer)
        print('-Error function: ')
        print(self.error_function)
        print('-Use ROI: %r' % self.use_roi)
        print('-Model to load: %s' % self.model_to_load)
        print('-Batch size: %d' % self.batch_size)
        print('-Shuffle Buffer: %s' % self.shuffle_buffer)
        print('-Validation size: %s' % self.val_size)
        print('-Max to models to keep: %d' % self.max_to_models_tokeep)

        self.experiment.close_logger()
    def run(self):
        self._create_graph()

        if self.experiment:
            self._log_setup()
            print = self.experiment.open_logger()

        print('###STARTING TRAINING####')
        if not self.model_to_load and not self.continue_session:
            print('Graph Variables initialized.')
            self.session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(self.logs_path, self.session.graph)

        # ---------------------------------------------------------------------

        # TRAIN LOOP------------------------------------------------------

        keep_training = True

        _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
        # summaries_iter = 1

        while keep_training:
            print(self.tfrecord_train)
            self.session.run(self.nodes_graph['iterator'].initializer,
                             feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            if self.val_size:
                print('usando validacion')
                self.session.run(self.nodes_graph['iterator_val'].initializer,
                                 feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_train})

            # np_input_batch =  self.session.run(self.nodes_graph['input_batch'])


            print(" epoch number " + str(current_epoch))
            while True:

                try:

                    if self.val_size:
                        _, loss, loss_val, train_summaries, val_summaries, step = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['loss_batch_val'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['val_summaries'],
                             self.nodes_graph['global_step']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})

                        # im,im_val = self.session.run(
                        #     [ self.nodes_graph['output_batch'], self.nodes_graph['output_batch_val']],
                        #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})
                        summary_writer.add_summary(train_summaries, step)
                        summary_writer.add_summary(val_summaries, step)
                    else:
                        _, loss, summaries, step = self.session.run(
                            [self.nodes_graph['train_step'], self.nodes_graph['loss_batch'],
                             self.nodes_graph['train_summaries'], self.nodes_graph['global_step']],
                            feed_dict={self.nodes_graph['learning_rate']: self.learning_rate,
                                       self.nodes_graph['is_training']: True})

                        # im = self.session.run(self.nodes_graph['output_batch']
                        #     ,
                        #     feed_dict={self.nodes_graph['learning_rate']: self.learning_rate, self.nodes_graph['is_training']: True})

                        summary_writer.add_summary(summaries, step)



                except tf.errors.OutOfRangeError:

                    model_dump_path = os.path.join(self.models_path, 'epoch_' + str(current_epoch) + ".ckpt")
                    save_path = self.saver.save(self.session, model_dump_path)
                    print("Model saved in file: %s" % save_path)



                    print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                    print(
                        " -----------------------------------EPOCH FINISHED-----------------------------------------------------------------")
                    if self.test:
                        self.session.run(self.nodes_graph['iterator_test'].initializer,
                                         feed_dict={self.nodes_graph['tfrecord_path']: self.tfrecord_test})

                        print(" epoch testing " + str(current_epoch))

                        loss_val_list = []

                        while True:

                            try:

                                loss_test = self.session.run(
                                    self.nodes_graph['loss_batch_test'])

                                loss_val_list.append(loss_test)



                            except tf.errors.OutOfRangeError:

                                test_loss_mean = np.mean(loss_val_list)
                                summary = tf.Summary()
                                summary.value.add(tag="avg_test_loss_epoch", simple_value=test_loss_mean)
                                summary_writer.add_summary(summary, current_epoch)
                                print('Test Loss: %f' % test_loss_mean)

                                print(time.strftime("%Y_%m_%d-%H_%M_%S"))
                                print(
                                    " -----------------------------------EPOCH testing FINISHED-----------------------------------------------------------------")

                                break

                    # current_epoch += 1
                    _, current_epoch = self.session.run([self.nodes_graph['update_epoch'], self.nodes_graph['epoch']])
                    if self.epochs is not None:
                        if current_epoch > self.epochs:
                            keep_training = False

                    break

        self.session.close()

        tf.reset_default_graph()
        print('TRAINING FINISHED.')
        if self.experiment:
            self.experiment.close_logger()




def get_input_from_record(record_path, batch_size, read_roi=False, val_size=None, buffer_shuffle=None, repeat = False, one_hot = False, labels = None):
    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):

        keys_to_features = dict()
        keys_to_features['rows_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['cols_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['channels_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['rows_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['cols_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['channels_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['network_in'] = tf.FixedLenFeature([], tf.string)
        keys_to_features['network_out'] = tf.FixedLenFeature([], tf.string)
        if read_roi:
            keys_to_features['rows_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['cols_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['channels_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['network_roi'] = tf.FixedLenFeature([], tf.string)

        parsed = tf.parse_single_example(record, keys_to_features)

        parsed_data = []

        rows_in = tf.cast(parsed['rows_in'], tf.int64)
        cols_in = tf.cast(parsed['cols_in'], tf.int64)
        channels_in = tf.cast(parsed['channels_in'], tf.int64)

        data_in = tf.decode_raw(parsed['network_in'], tf.float32)
        data_in = tf.reshape(data_in, tf.stack([rows_in, cols_in, channels_in]))
        data_in = tf.cast(data_in, tf.float32)

        parsed_data.append(data_in)

        rows_out = tf.cast(parsed['rows_out'], tf.int64)
        cols_out = tf.cast(parsed['cols_out'], tf.int64)
        channels_out = tf.cast(parsed['channels_out'], tf.int64)

        data_out = tf.decode_raw(parsed['network_out'], tf.float32)
        data_out = tf.reshape(data_out, tf.stack([rows_out, cols_out, channels_out]))
        data_out = tf.cast(data_out, tf.float32)
        if one_hot:
            data_out = tf.one_hot(data_out,labels)


        parsed_data.append(data_out)
        if read_roi:
            rows_roi = tf.cast(parsed['rows_roi'], tf.int64)
            cols_roi = tf.cast(parsed['cols_roi'], tf.int64)
            channels_roi = tf.cast(parsed['channels_roi'], tf.int64)

            data_roi = tf.decode_raw(parsed['network_roi'], tf.float32)
            data_roi = tf.reshape(data_roi, tf.stack([rows_roi, cols_roi, channels_roi]))
            data_roi = tf.cast(data_roi, tf.float32)
            parsed_data.append(data_roi)

        return parsed_data

    full_dataset = dataset.map(parser)


    if buffer_shuffle:
        full_dataset = full_dataset.shuffle(buffer_size=buffer_shuffle)
    full_dataset = full_dataset.batch(batch_size)

    if val_size:
        val_dataset = full_dataset.take(val_size)
        val_dataset = val_dataset.repeat()
        train_dataset = full_dataset.skip(val_size)


        train_dataset = train_dataset.repeat(1)
        train_dataset.prefetch(buffer_size=buffer_shuffle)

        iterator_train = train_dataset.make_initializable_iterator()
        iterator_val = val_dataset.make_initializable_iterator()


        data_train = iterator_train.get_next()


        data_val = iterator_val.get_next()

        return (*data_train), iterator_train, (*data_val), iterator_val
    elif repeat:
        full_dataset = full_dataset.repeat()
        full_dataset.prefetch(buffer_size=buffer_shuffle)
        iterator = full_dataset.make_initializable_iterator()

        data = iterator.get_next()
    elif not repeat:
        full_dataset = full_dataset.repeat(1)
        full_dataset.prefetch(buffer_size=buffer_shuffle)
        iterator = full_dataset.make_initializable_iterator()

        data = iterator.get_next()
    return (*data), iterator

def get_input_from_dataset(dataset):
    iterator = dataset.make_initializable_iterator()
    data = iterator.get_next()
    return (*data), iterator

def get_input_from_record_patch3d(record_path, batch_size, read_roi=False, val_size=None, buffer_shuffle=None, repeat = False):
    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):

        keys_to_features = dict()
        keys_to_features['rows_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['cols_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['depth_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['channels_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['rows_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['cols_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['depth_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['channels_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['network_in'] = tf.FixedLenFeature([], tf.string)
        keys_to_features['network_out'] = tf.FixedLenFeature([], tf.string)
        if read_roi:
            keys_to_features['rows_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['cols_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['depth_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['channels_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['network_roi'] = tf.FixedLenFeature([], tf.string)

        parsed = tf.parse_single_example(record, keys_to_features)

        parsed_data = []

        rows_in = tf.cast(parsed['rows_in'], tf.int64)
        cols_in = tf.cast(parsed['cols_in'], tf.int64)
        depth_in = tf.cast(parsed['depth_in'], tf.int64)
        channels_in = tf.cast(parsed['channels_in'], tf.int64)

        data_in = tf.decode_raw(parsed['network_in'], tf.float32)
        data_in = tf.reshape(data_in, tf.stack([rows_in, cols_in,depth_in, channels_in]))
        data_in = tf.cast(data_in, tf.float32)

        parsed_data.append(data_in)

        rows_out = tf.cast(parsed['rows_out'], tf.int64)
        cols_out = tf.cast(parsed['cols_out'], tf.int64)
        depth_out = tf.cast(parsed['depth_out'], tf.int64)
        channels_out = tf.cast(parsed['channels_out'], tf.int64)

        data_out = tf.decode_raw(parsed['network_out'], tf.float32)
        data_out = tf.reshape(data_out, tf.stack([rows_out, cols_out, depth_out,channels_out]))
        data_out = tf.cast(data_out, tf.float32)
        parsed_data.append(data_out)
        if read_roi:
            rows_roi = tf.cast(parsed['rows_roi'], tf.int64)
            cols_roi = tf.cast(parsed['cols_roi'], tf.int64)
            depth_roi = tf.cast(parsed['depth_roi'], tf.int64)
            channels_roi = tf.cast(parsed['channels_roi'], tf.int64)

            data_roi = tf.decode_raw(parsed['network_roi'], tf.float32)
            data_roi = tf.reshape(data_roi, tf.stack([rows_roi, cols_roi, depth_roi,channels_roi]))
            data_roi = tf.cast(data_roi, tf.float32)
            parsed_data.append(data_roi)

        return parsed_data

    full_dataset = dataset.map(parser)
    if buffer_shuffle:
        full_dataset = full_dataset.shuffle(buffer_size=buffer_shuffle)
    full_dataset = full_dataset.batch(batch_size)

    if val_size:
        val_dataset = full_dataset.take(val_size)
        val_dataset = val_dataset.repeat()
        train_dataset = full_dataset.skip(val_size)

        train_dataset = train_dataset.repeat(1)
        iterator_train = train_dataset.make_initializable_iterator()
        iterator_val = val_dataset.make_initializable_iterator()

        data_train = iterator_train.get_next()

        data_val = iterator_val.get_next()

        return (*data_train), iterator_train, (*data_val), iterator_val
    elif repeat:
        full_dataset = full_dataset.repeat()

        iterator = full_dataset.make_initializable_iterator()

        data = iterator.get_next()
    elif not repeat:
        full_dataset = full_dataset.repeat(1)

        iterator = full_dataset.make_initializable_iterator()

        data = iterator.get_next()
    return (*data), iterator




def get_input_from_record3d2d(record_path, batch_size, read_roi=False, val_size=None, buffer_shuffle=None):
    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):

        keys_to_features = dict()
        keys_to_features['rows_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['cols_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['channels_in'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['rows_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['cols_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['channels_out'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['network_in'] = tf.FixedLenFeature([], tf.string)
        keys_to_features['network_out'] = tf.FixedLenFeature([], tf.string)
        keys_to_features['deep'] = tf.FixedLenFeature([], tf.int64)
        keys_to_features['vols'] = tf.FixedLenFeature([], tf.string)
        keys_to_features['slice_id'] = tf.FixedLenFeature([], tf.int64)


        if read_roi:
            keys_to_features['rows_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['cols_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['channels_roi'] = tf.FixedLenFeature([], tf.int64)
            keys_to_features['network_roi'] = tf.FixedLenFeature([], tf.string)

        parsed = tf.parse_single_example(record, keys_to_features)

        parsed_data = []




        rows_in = tf.cast(parsed['rows_in'], tf.int64)
        cols_in = tf.cast(parsed['cols_in'], tf.int64)
        channels_in = tf.cast(parsed['channels_in'], tf.int64)

        data_in = tf.decode_raw(parsed['network_in'], tf.float32)
        data_in = tf.reshape(data_in, tf.stack([rows_in, cols_in, channels_in]))
        data_in = tf.cast(data_in, tf.float32)

        parsed_data.append(data_in)



        rows_out = tf.cast(parsed['rows_out'], tf.int64)
        cols_out = tf.cast(parsed['cols_out'], tf.int64)
        channels_out = tf.cast(parsed['channels_out'], tf.int64)

        data_out = tf.decode_raw(parsed['network_out'], tf.float32)
        data_out = tf.reshape(data_out, tf.stack([rows_out, cols_out, channels_out]))
        data_out = tf.cast(data_out, tf.float32)
        parsed_data.append(data_out)

        vols =  tf.decode_raw(parsed['vols'], tf.float32)
        deep = tf.cast(parsed['deep'], tf.int64)
        vols = tf.reshape(vols, tf.stack([deep, 128, 128, 6]))
        slice_id = tf.cast(parsed['slice_id'], tf.int64)
        parsed_data.append(vols)
        parsed_data.append(slice_id)


        if read_roi:
            rows_roi = tf.cast(parsed['rows_roi'], tf.int64)
            cols_roi = tf.cast(parsed['cols_roi'], tf.int64)
            channels_roi = tf.cast(parsed['channels_roi'], tf.int64)

            data_roi = tf.decode_raw(parsed['network_roi'], tf.float32)
            data_roi = tf.reshape(data_roi, tf.stack([rows_roi, cols_roi, channels_roi]))
            data_roi = tf.cast(data_roi, tf.float32)
            parsed_data.append(data_roi)

        return parsed_data

    full_dataset = dataset.map(parser)
    if buffer_shuffle:
        full_dataset = full_dataset.shuffle(buffer_size=buffer_shuffle)
    full_dataset = full_dataset.batch(batch_size)

    if val_size:
        val_dataset = full_dataset.take(val_size)
        val_dataset = val_dataset.repeat()
        train_dataset = full_dataset.skip(val_size)

        train_dataset = train_dataset.repeat(1)
        iterator_train = train_dataset.make_initializable_iterator()
        iterator_val = val_dataset.make_initializable_iterator()

        data_train = iterator_train.get_next()

        data_val = iterator_val.get_next()

        return (*data_train), iterator_train, (*data_val), iterator_val
    else:
        full_dataset = full_dataset.repeat(1)

        iterator = full_dataset.make_initializable_iterator()

        data = iterator.get_next()

        return (*data), iterator



def tensorboard_summaries(train_scalars_dict, train_images_dict, val_scalars_dict, val_images_dict):
    def create_scalar_summary(dict):

        summary = []
        for name, tensor in zip(dict.keys(), dict.values()):
            summary.append(tf.summary.scalar(name=name, tensor=tensor))

        return summary

    def create_image_summary(dict):

        summary = []
        for name, tensor in zip(dict.keys(), dict.values()):
            summary.append(tf.summary.image(name=name, tensor=tensor))

        return summary

    train_summaries = tf.summary.merge(
        create_scalar_summary(train_scalars_dict) + create_image_summary(train_images_dict))

    val_summaries = tf.summary.merge(
        create_scalar_summary(val_scalars_dict) + create_image_summary(val_images_dict))

    return train_summaries, val_summaries


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.


    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    output = tf.cast(output, tf.float32)
    target = tf.cast(target, tf.float32)
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice