import tensorflow as tf

class Network_DeepLab_v4_Xception:
    def __init__(self, name, input, input_channels, is_training=True, reuse=None, filters=8, deep=1, n_classes=2,
                 batch_size=None, second_stride_unit=False, dropout=False, dropout_prob=0.5,
                 renorm=False):
        self.imgs = input
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
        self.momentum = 0.99
        self.renorm = renorm
        self.output = self.build_net()

    def conv_init(self, x, filters):

        with tf.variable_scope('conv_init_3x3_stride_conv_3x3', reuse=self.reuse):
            print('conv_init_3x3_stride + conv_3x3')

            x = tf.layers.conv2d(
                x, filters, [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')
        return x

    def entry_flow_block(self, x, filters, dilatation):

        self.entry_flow_counter += 1

        with tf.variable_scope('Entry_flow_block_%d' % self.entry_flow_counter, reuse=self.reuse):
            print('Entry flow block %d' % self.entry_flow_counter)

            x_res = tf.layers.conv2d(
                x, filters, [1, 1], padding='SAME', dilation_rate=dilatation)

            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn3')
            x = tf.nn.relu(x + x_res, name='relu3')

        return x

    def middle_flow_block(self, x, filters):

        self.middle_flow_counter += 1

        with tf.variable_scope('middle_flow_block_%d' % self.middle_flow_counter, reuse=self.reuse):
            print('middle flow block %d' % self.middle_flow_counter)
            x_res = x
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')
            x = tf.nn.relu(x + x_res, name='relu3')

        return x

    def final_flow_block(self, x, filters, dilatation):

        with tf.variable_scope('final_flow_lock', reuse=self.reuse):
            print('final flow block')

            x_res = tf.layers.conv2d(
                x, 2 * filters, [1, 1], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')

            x = tf.nn.relu(x, name='relu2')

            x = tf.layers.separable_conv2d(x, 2 * filters, [3, 3], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')

            x = tf.nn.relu(x + x_res, name='relu3')

            x = tf.layers.separable_conv2d(
                x, 2 * filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn4')
            x = tf.nn.relu(x, name='relu4')
            x = tf.layers.separable_conv2d(
                x, 2 * filters, [3, 3], padding='SAME', dilation_rate=2 * dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn5')
            x = tf.nn.relu(x, name='relu5')

            x = tf.layers.separable_conv2d(
                x, 2 * filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn6')
            x = tf.nn.relu(x, name='relu6')

        return x

    def conv_unit(self, x, filters):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x_orig = x

            x = tf.layers.conv2d(x, filters / 4, [1, 1], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu')

            x = tf.layers.conv2d(x, filters / 4, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu')

            x = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn3')

            # if tf.shape(x)[3] != tf.shape(x_orig)[3]:
            #     x_orig = self.conv1x1(x_orig, filters)

            x = tf.nn.relu(x + x_orig, name='relu')

            return x

    def atrous_unit(self, x, filters, dilatation):
        self.atrous_counter += 1
        with tf.variable_scope('atrous_unit_%d' % self.atrous_counter, reuse=self.reuse):
            print('atrous_unit_%d' % self.atrous_counter)

            x_orig = x

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x + x_orig, name='relu1')

            return x

    def aspp(self, x, filters):
        self.aspp_counter += 1
        with tf.variable_scope('aspp_unit_%d' % self.aspp_counter, reuse=self.reuse):
            print('aspp_unit_%d' % self.aspp_counter)

            x_1 = self.atrous_unit(x, filters, 4)  # 4
            x_2 = self.atrous_unit(x, filters, 8)  # 8
            x_3 = self.atrous_unit(x, filters, 16)  # 16
            x_4 = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')
            x_5 = tf.layers.average_pooling2d(x, [2, 2], [1, 1], padding='SAME')
            # shape_orig = tf.shape(x)
            # shape_pool = tf.shape(x_5)
            # x_5 = tf.pad(x_5, [[0, 0], [shape_orig[1] - shape_pool[1], 0], [shape_orig[2] - shape_pool[2], 0], [0, 0]])
            #
            # x_5 = tf.reshape(x_5, tf.shape(x))

            x = tf.concat([x_1, x_2, x_3, x_4, x_5], 3)
            x = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn')
            x = tf.nn.relu(x, name='relu')
            return x

    def decoder_flow(self, x1, x2, x3, filters):

        with tf.variable_scope('Decoder_flow', reuse=self.reuse):
            print('Decoder_flow')
            x = tf.concat([x1, x2, x3], 3)
            x = tf.layers.conv2d_transpose(
                x, int(filters / 4), [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            x = self.middle_flow_block(x, int(filters / 4))
            x = tf.layers.conv2d(x, 1, [1, 1], padding='SAME')
            return x

    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):

            x = self.imgs


            x.set_shape([None, None, None, self.input_channels])


            x = self.conv_init(x, self.filters)

            for _, dilatation in enumerate([2, 4]):
                x = self.entry_flow_block(x, dilatation * self.filters, dilatation)

            entry2_decoder = x

            for middle_block in range(self.deep):
                x = self.middle_flow_block(x, 4 * self.filters)

            x = self.final_flow_block(x, 4 * self.filters, dilatation=8)

            entry3_decoder = x

            for _ in range(2):

                x = self.aspp( x, 2*4*self.filters)

            x = self.decoder_flow(x, entry2_decoder, entry3_decoder, 4 * self.filters)

            return x


class Network_DeepLab_v4_Xception2:
    def __init__(self, name, input, input_channels, is_training=True, reuse=None, filters=8, middle_blocks=1, strides_blocks = 1, dilatation_blocks = 2,aspp_blocks = 2, n_classes=2,
                 batch_size=None, second_stride_unit=False, dropout=False, dropout_prob=0.5,
                 renorm=False):
        self.imgs = input
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.filters = filters
        self.middle_blocks = middle_blocks
        self.n_classes = n_classes
        self.second_stride_unit = second_stride_unit
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.strides_blocks = strides_blocks
        self.aspp_blocks = aspp_blocks
        self.dilatation_blocks = dilatation_blocks
        self.entry_flow_counter = 0
        self.middle_flow_counter = 0
        self.atrous_counter = 0
        self.aspp_counter = 0
        self.conv_counter = 0
        self.momentum = 0.99
        self.renorm = renorm
        self.output = self.build_net()

    def conv_init(self, x, filters):

        with tf.variable_scope('conv_init_3x3_stride_conv_3x3', reuse=self.reuse):
            print('conv_init_3x3_stride + conv_3x3')

            x = tf.layers.conv2d(
                x, filters, [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.conv2d(x,2* filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')
        return x

    def entry_flow_block_stride(self, x, filters):

        self.entry_flow_counter += 1

        with tf.variable_scope('Entry_flow_block_%d' % self.entry_flow_counter, reuse=self.reuse):
            print('Entry flow block %d' % self.entry_flow_counter)


            x_res = tf.layers.conv2d(
                x, filters, [1, 1], padding='SAME', strides=(2, 2))

            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn3')
            x = tf.nn.relu(x + x_res, name='relu3')

        return x

    def entry_flow_block_dilatation(self, x, filters, dilatation):

        self.entry_flow_counter += 1

        with tf.variable_scope('Entry_flow_block_%d' % self.entry_flow_counter, reuse=self.reuse):
            print('Entry flow block %d' % self.entry_flow_counter)

            x_res = tf.layers.conv2d(
                x, filters, [1, 1], padding='SAME', dilation_rate=dilatation)

            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn3')
            x = tf.nn.relu(x + x_res, name='relu3')

        return x

    def middle_flow_block(self, x, filters):

        self.middle_flow_counter += 1

        with tf.variable_scope('middle_flow_block_%d' % self.middle_flow_counter, reuse=self.reuse):
            print('middle flow block %d' % self.middle_flow_counter)
            x_res = x
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu2')
            x = tf.layers.separable_conv2d(x, filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')
            x = tf.nn.relu(x + x_res, name='relu3')

        return x

    def final_flow_block(self, x, filters, dilatation):

        with tf.variable_scope('final_flow_lock', reuse=self.reuse):
            print('final flow block')

            x_res = tf.layers.conv2d(
                x,  filters/2, [1, 1], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.separable_conv2d(x, filters/4, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.separable_conv2d(x, filters/2, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')

            x = tf.nn.relu(x, name='relu2')

            x = tf.layers.separable_conv2d(x, filters/2, [3, 3], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')

            x = tf.nn.relu(x + x_res, name='relu3')

            x = tf.layers.separable_conv2d(
                x, int(filters*3/4 ), [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn4')
            x = tf.nn.relu(x, name='relu4')
            x = tf.layers.separable_conv2d(
                x, int(filters*3/4 ), [3, 3], padding='SAME', dilation_rate=2 * dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn5')
            x = tf.nn.relu(x, name='relu5')

            x = tf.layers.separable_conv2d(
                x,  filters, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn6')
            x = tf.nn.relu(x, name='relu6')

        return x

    def conv_unit(self, x, filters):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x_orig = x

            x = tf.layers.conv2d(x, filters / 4, [1, 1], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu')

            x = tf.layers.conv2d(x, filters / 4, [3, 3], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn2')
            x = tf.nn.relu(x, name='relu')

            x = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn3')

            # if tf.shape(x)[3] != tf.shape(x_orig)[3]:
            #     x_orig = self.conv1x1(x_orig, filters)

            x = tf.nn.relu(x + x_orig, name='relu')

            return x

    def atrous_unit(self, x, filters, dilatation):
        self.atrous_counter += 1
        with tf.variable_scope('atrous_unit_%d' % self.atrous_counter, reuse=self.reuse):
            print('atrous_unit_%d' % self.atrous_counter)

            x_orig = x

            x = tf.layers.separable_conv2d(
                x, filters, [3, 3], padding='SAME', dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x + x_orig, name='relu1')

            return x

    def aspp(self, x, filters):
        self.aspp_counter += 1
        with tf.variable_scope('aspp_unit_%d' % self.aspp_counter, reuse=self.reuse):
            print('aspp_unit_%d' % self.aspp_counter)

            x_1 = self.atrous_unit(x, filters, 4)  # 4
            x_2 = self.atrous_unit(x, filters, 8)  # 8
            x_3 = self.atrous_unit(x, filters, 16)  # 16
            x_4 = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')
            x_5 = tf.layers.average_pooling2d(x, [2, 2], [1, 1], padding='SAME')
            # shape_orig = tf.shape(x)
            # shape_pool = tf.shape(x_5)
            # x_5 = tf.pad(x_5, [[0, 0], [shape_orig[1] - shape_pool[1], 0], [shape_orig[2] - shape_pool[2], 0], [0, 0]])
            #
            # x_5 = tf.reshape(x_5, tf.shape(x))

            x = tf.concat([x_1, x_2, x_3, x_4, x_5], 3)
            x = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn')
            x = tf.nn.relu(x, name='relu')
            return x

    def decoder_flow1(self, x1, x2, x3, filters, strides_blocks ):
        with tf.variable_scope('Decoder_flow', reuse=self.reuse):
            x = tf.concat([x1, x2, x3], 3)
            half_filters =  2
            print('Decoder_flow')
            for stride_iter in range(strides_blocks):
                with tf.variable_scope('Decoder_flow_iter_%d' % stride_iter, reuse=self.reuse):

                    x = tf.layers.conv2d_transpose(
                        x, int(filters / half_filters), [3, 3], padding='SAME', strides=(2, 2))
                    x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                                      momentum=self.momentum, renorm=self.renorm, name='bn1')
                    x = tf.nn.relu(x, name='relu1')

                    x = self.middle_flow_block(x, int(filters / half_filters))

                    half_filters = half_filters * 2

            x = tf.layers.conv2d(x, 1, [1, 1], padding='SAME')
            return x

    def decoder_flow2(self, x1, x2, x3, filters, ):

        with tf.variable_scope('Decoder_flow', reuse=self.reuse):
            print('Decoder_flow')
            x = tf.concat([x1, x2, x3], 3)
            x = tf.layers.conv2d_transpose(
                x, int(filters / 4), [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            x = self.middle_flow_block(x, int(filters / 4))
            x = tf.layers.conv2d(x, 1, [1, 1], padding='SAME')
            return x
    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):

            x = self.imgs

            filters_track = self.filters

            x.set_shape([None, None, None, self.input_channels])


            x = self.conv_init(x, self.filters)

            filters_track = 2* filters_track

            for _ in range(self.strides_blocks):
                filters_track = 2*filters_track
                x = self.entry_flow_block_stride(x, filters_track)

            dilatation_track = 2
            for _ in range(self.dilatation_blocks):
                filters_track = 2*filters_track
                x = self.entry_flow_block_dilatation(x,filters_track, dilatation = dilatation_track)
                dilatation_track = 2* dilatation_track
            entry2_decoder = x

            for middle_block in range(self.middle_blocks):
                x = self.middle_flow_block(x, filters_track)

            filters_track = 4 * filters_track
            x = self.final_flow_block(x, filters_track, dilatation=dilatation_track)

            entry3_decoder = x

            for _ in range(self.aspp_blocks):

                x = self.aspp( x, filters_track)

            x = self.decoder_flow1(x, entry2_decoder, entry3_decoder, filters_track, strides = self.strides_blocks + 1)

            return x

