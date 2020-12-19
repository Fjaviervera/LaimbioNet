import tensorflow as tf


class No_pool_net_3d:

    def __init__(self, name, input, input_channels, output_channels = 1, is_training=True, reuse=None, filters=32, deep=1, n_classes=2,
                 batch_size=None, second_stride_unit=False, dropout=False, dropout_prob=0.5,
                 renorm=False):
        self.imgs = input
        self.input_channels = input_channels
        self.output_channels = output_channels
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
        self.momentum = 0.99
        self.renorm = renorm
        self.output = self.build_net()

    def __str__(self):
        return ('U-net estandar designed by javier.vera@urjc.es')


    def conv(self, x, filters, kernel = 3, stride=1, dilatation = 1):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x = tf.layers.conv3d(
                x, filters, [kernel, kernel,kernel], padding='SAME', strides=(stride, stride,stride),dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def deconv(self, x, filters, stride=2):
        self.deconv_counter += 1

        with tf.variable_scope('deconv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('deconv_unit_%d' % self.conv_counter)

            x = tf.layers.conv2d_transpose(
                x, filters, [3, 3], padding='SAME', strides=(2, 2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def n_convs(self, x, n, filters, kernel = 3, dilatation = 1):
        for _ in range(n):
            x = self.conv(x, filters, kernel = kernel, dilatation = dilatation)
        return x

    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):


            x = self.imgs


            x.set_shape([None, 32, 32,32, self.input_channels])

            x_res = x
            orig_shapes = [tf.shape(x)[1], tf.shape(x)[2]]
            filters_num = self.filters

            x = self.n_convs(x, 1, self.filters, kernel = 5)


            x = self.n_convs(x, 3, 2*self.filters)

            x = self.n_convs(x, 1, 2*self.filters,kernel=5)

            x = self.n_convs(x, 2, 2*self.filters)

            x = self.n_convs(x, 1,  self.filters, kernel=5)

            x = self.n_convs(x, 1, self.filters, kernel=3)

            with tf.variable_scope('final_conv' , reuse=self.reuse):
                print('final_conv')

                x = tf.layers.conv3d(
                    x, self.output_channels, [1, 1,1], padding='SAME', strides=(1, 1,1), dilation_rate=1)

            return x

