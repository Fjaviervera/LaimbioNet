
import tensorflow as tf


class network_3d_2d:

    def __init__(self, name, input_channels, vol_input, imgs_input, slice_id, is_training=True, reuse=None, filters=8, deep=2, n_classes=2,
                 batch_size=None, second_stride_unit=False, dropout=False, dropout_prob=0.5,
                 renorm=False):
        self.imgs_input = imgs_input
        self.vol_input = vol_input
        self.slice_id = slice_id
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.filters = filters
        self.deep = deep
        self.n_classes = n_classes
        self.second_stride_unit = second_stride_unit
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.entry_flow_counter = 0
        self.middle_flow_counter = 0
        self.atrous_counter = 0
        self.aspp_counter = 0
        self.conv_counter = 0
        self.deconv_counter = 0
        self.conv_counter_2d = 0
        self.deconv_counter_2d = 0
        self.sep_conv3d_counter = 0
        self.momentum = 0.99
        self.renorm = renorm
        self.output = self.build_net()

    def conv(self, x, filters, stride=1):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x = tf.layers.conv3d(
                x, filters, [3, 3, 3], padding='SAME', strides=(1, stride, stride))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def conv_stride3d(self, x, filters, stride=1):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x = tf.layers.conv3d(
                x, filters, [3, 3, 3], padding='SAME', strides=(1, stride, stride))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def deconv(self, x, filters, stride=2):
        self.deconv_counter += 1

        with tf.variable_scope('deconv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('deconv_unit_%d' % self.conv_counter)

            x = tf.layers.conv3d_transpose(
                x, filters, [3, 3, 3], padding='SAME', strides=(stride, stride, stride))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def n_convs(self, x, n, filters):
        for _ in range(n):
            x = self.conv(x, filters)
        return x

    def conv_2d(self, x, filters, stride=1):
        self.conv_counter_2d += 1

        with tf.variable_scope('conv_unit2d_%d' % self.conv_counter_2d, reuse=self.reuse):
            print('conv_unit2d_%d' % self.conv_counter_2d)

            x = tf.layers.conv2d(
                x, filters, [3, 3], padding='SAME', strides=(stride, stride))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def deconv_2d(self, x, filters, stride=2):
        self.deconv_counter_2d += 1

        with tf.variable_scope('deconv_unit2d_%d' % self.deconv_counter_2d, reuse=self.reuse):
            print('deconv_unit2d_%d' % self.deconv_counter_2d)

            x = tf.layers.conv2d_transpose(
                x, filters, [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def n_convs_2d(self, x, n, filters):
        for _ in range(n):
            x = self.conv_2d(x, filters)
        return x

    def slicing(self, x, slice_input):
        sliced_x = x[0, slice_input[0],:, :,:]
        sliced_x = tf.expand_dims(sliced_x, 0)

        for slice_i in range(1, self.batch_size):
            sliced_x = tf.concat([sliced_x, tf.expand_dims(x[slice_i, slice_input[slice_i],:,:, :], 0)], 0)
        return sliced_x

    def slicing_assign(self, x, slice_input):

        sliced_x = tf.Variable(tf.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[4]]
                                        , dtype=tf.float32), trainable=False)

        for slice_i in range(slice_input.shape[0]):
            sliced_x[slice_i, :, :, :].assign(x[slice_i, :, :, slice_input[slice_i, 0], :])
        return sliced_x

    def separable_conv3d(self, x, filters):
        self.sep_conv3d_counter += 1

        with tf.variable_scope('separable_3d_conv_%d' % self.sep_conv3d_counter, reuse=self.reuse):
            print('separable_3d_conv_%d' % self.sep_conv3d_counter)

            channels_out = tf.Variable(tf.zeros([1,-1, x.shape[-3], x.shape[-2], x.shape[-1]] , dtype=tf.float32), validate_shape=True,trainable=False)

            # x = tf.split(x, x.shape[-1], axis= -1)
            #
            #
            # for channel_id,channel_x in enumerate(x):
            #     x[channel_id] = tf.layers.conv3d(
            #           channel_x, 1, [3, 3, 3], padding='SAME' )
            #
            # x =tf.squeeze(tf.stack(x,axis = -1), axis = -2)

            # channels_out = tf.get_variable(name='sep_conv_var',shape = tf.shape(x), initializer=tf.zeros_initializer(), dtype=tf.float32, validate_shape=False, trainable=False)
            for channel_i in range(x.shape[-1]):

                channels_out[:, :, :, :, channel_i].assign(tf.layers.conv3d(
                    tf.expand_dims(x[:, :, :, :, channel_i], -1), 1, [3, 3, 3], padding='SAME')
                )

            x = tf.reshape(channels_out, shape=[-1,385,x.shape[-3], x.shape[-2], x.shape[-1]])

            x = tf.layers.conv3d(
               x, filters, [1, 1, 1], padding='SAME')

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')

            return x

    def n_sepconvs_3d(self, x, n, filters):
        for _ in range(n):
            x = self.separable_conv3d(x, filters)
        return x

    #
    # def build_net(self):
    #     print('Building Network:')
    #     print(self.name)
    #     with tf.variable_scope(self.name, reuse=self.reuse):
    #         # 3D processing path
    #
    #
    #
    #
    #         x1 = self.vol_input
    #         x2 = self.imgs_input
    #         x1.set_shape([None, None,None, None, 6])
    #         x2.set_shape([None, None, None, self.input_channels])
    #
    #
    #         # x2 = self.n_convs_2d(x2, 8, 64 * self.filters)
    #         #
    #         # x2= tf.layers.conv2d(x2, 1, [1, 1], padding='SAME')
    #         # return x2
    #         # x1 = tf.squeeze(x1, axis = [3])
    #         #
    #         # x2 = self.slice_input
    #         # x2 = self.conv_2d(x2, self.filters, stride = 2)
    #         # x2 = self.n_convs_2d(x2, 2,self.filters)
    #         # x2 = self.conv_2d(x2,2* self.filters, stride = 2)
    #         # x2 = self.n_convs_2d(x2, 4,2 * self.filters)
    #         #
    #         # x1 = tf.concat([x1,x2],3)
    #         #
    #         # x1 = self.deconv_2d(x1,2*self.filters)
    #         #
    #         # x1 = self.n_convs_2d(x1, 4,2*self.filters)
    #         #
    #         # x1 = self.deconv_2d(x1,self.filters)
    #         #
    #         # x1 = self.n_convs_2d(x1, 4,self.filters)
    #         #
    #         # x1 = tf.layers.conv2d(x1, 1, [1, 1], padding='SAME')
    #         # return x1
    #



    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):
            # 3D processing path




            x1 = self.vol_input
            x2 = self.imgs_input
            x1.set_shape([None, None,None, None, 6])
            x2.set_shape([None, None, None, self.input_channels])



            slice_1 = x2
            # x1 = self.conv(x1, self.filters, stride = 2)
            x1 = self.n_convs(x1, 3, 2 *self.filters)
            slice_2 = self.slicing(x1, self.slice_id)
            x1 = self.conv(x1,2* self.filters, stride = 2)
            x1 = self.n_convs(x1, 4,4 * self.filters)
            slice_3 = self.slicing(x1, self.slice_id)
            x1 = self.conv(x1,4* self.filters, stride = 2)
            x1 = self.n_convs(x1, 4,8 * self.filters)
            slice_4 = self.slicing(x1, self.slice_id)

            # x1 = self.slicing(self.vol_input, self.slice_input)
            # slice_1 = self.slice_input
            # x1 = self.n_convs_2d(self.vol_input, 2, self.filters)
            # x1 = self.conv_2d(x1, 2 * self.filters, stride=2)
            # x1 = self.n_convs_2d(x1, 2, self.filters)
            # slice_2 = x1
            # x1 = self.conv_2d(x1, 4 * self.filters, stride=2)
            # x1 = self.n_convs_2d(x1, 3, 4 * self.filters)
            # slice_3 = x1
            # x1 = self.conv_2d(x1, 8 * self.filters, stride=2)
            # x1 = self.n_convs_2d(x1, 4, 8 * self.filters)
            # slice_4 = x1
            # x1 = self.conv_2d(x1, 16 * self.filters, stride=2)
            # x1 = self.n_convs_2d(x1, 5, 16 * self.filters)
            # slice_5 = x1

            # slice_1 = self.slicing(x1, self.slice_input)
            # x1 = self.conv(x1, self.filters, stride = 2)
            # x1 = self.n_sepconvs_3d(x1, 2,self.filters)
            # slice_2 = self.slicing(x1, self.slice_input)
            # x1 = self.conv(x1,2* self.filters, stride = 2)
            # x1 = self.n_sepconvs_3d(x1, 3,2 * self.filters)
            # slice_3 = self.slicing(x1, self.slice_input)
            # x1 = self.conv(x1,4* self.filters, stride = 2)
            # x1 = self.n_sepconvs_3d(x1, 4,4 * self.filters)
            # slice_4 = self.slicing(x1, self.slice_input)



            # x1 = self.deconv_2d(slice_5, 8 * self.filters)
            # x1 = self.n_convs_2d(x1, 5, 8 * self.filters)
            # x1 = tf.concat([x1, slice_4], 3)

            x1 = self.deconv_2d(slice_4, 4 * self.filters)
            x1 = tf.concat([x1, slice_3], 3)
            x1 = self.n_convs_2d(x1, 4, 4 * self.filters)

            x1 = self.deconv_2d(x1, 2* self.filters)
            x1 = tf.concat([x1, slice_2], 3)
            x1 = self.n_convs_2d(x1, 2, 2 * self.filters)

            x1 = self.deconv_2d(x1, self.filters)
            x1 = tf.concat([x1, slice_1], 3)
            x1 = self.n_convs_2d(x1, 4, self.filters)

            x1 = tf.layers.conv2d(x1, 1, [1, 1], padding='SAME')
            return x1
            # x1 = tf.squeeze(x1, axis = [3])

            # x2 = self.slice_input
            # x2 = self.conv_2d(x2, self.filters, stride = 2)
            # x2 = self.n_convs_2d(x2, 2,self.filters)
            # x2 = self.conv_2d(x2,2* self.filters, stride = 2)
            # x2 = self.n_convs_2d(x2, 4,2 * self.filters)

            # x1 = tf.concat([x1,x2],3)

            # x1 = self.deconv_2d(x1,2*self.filters)

            # x1 = self.n_convs_2d(x1, 4,2*self.filters)

            # x1 = self.deconv_2d(x1,self.filters)

            # x1 = self.n_convs_2d(x1, 4,self.filters)

            # x1 = tf.layers.conv2d(x1, 1, [1, 1], padding='SAME')
            # return x1




