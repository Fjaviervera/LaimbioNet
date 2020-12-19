import tensorflow as tf


class Unet3d_residuals:

    def __init__(self, name, input, input_channels, output_channels = 1, is_training=True, reuse=None, filters=32, n_classes=2,
                 batch_size=None, dropout=False, dropout_prob=0.5,
                 renorm=False, input_shape = (32,32,32)):
        self.imgs = input
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.batch_size = batch_size
        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.filters = filters
        self.n_classes = n_classes
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.input_shape = input_shape
        self.xception_block_counter = 0

        self.conv_counter = 0
        self.deconv_counter = 0
        self.momentum = 0.99
        self.renorm = renorm
        self.output = self.build_net()

    def __str__(self):
        return ('U-net estandar designed by javier.vera@urjc.es')

    def xception_block(self, inputs, filters, kernel, last_stride=1, dilatation=1, do_residual=True):

        self.xception_block_counter += 1

        with tf.variable_scope('Xception_block_counter_%d' % self.xception_block_counter, reuse=self.reuse):
            print('Xception block counter %d' % self.xception_block_counter)

            if do_residual:

                if last_stride != 1 or int(inputs.shape[3]) != filters:
                    x_res = tf.nn.relu(inputs, name='relu_input_residual')

                    x_res = tf.layers.conv3d(
                        x_res, filters, 1, padding='SAME', strides=last_stride, dilation_rate=1)

                    x_res = tf.layers.batch_normalization(x_res, training=self.is_training, reuse=self.reuse,
                                                          momentum=self.momentum, renorm=self.renorm,
                                                          name='bn_residual')
                else:
                    x_res = inputs

            else:
                x_res = inputs

            # x = tf.nn.relu(inputs, name='relu_input')
            x = tf.layers.conv3d(
                inputs, filters, kernel, padding='SAME', strides=1, dilation_rate=1)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x, name='relu_1')
            x = tf.layers.conv3d(
                x, filters, kernel, padding='SAME', strides=1, dilation_rate=1)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')

            x = tf.nn.relu(x, name='relu_2')
            #
            # x = tf.layers.conv3d(
            #     x, filters, kernel, padding='SAME', strides=last_stride, dilation_rate=dilatation)
            #
            # x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
            #                                   momentum=self.momentum, renorm=self.renorm, name='bn3')

            if do_residual:
                x = x + x_res

            return x

    def conv(self, x, filters, stride=1):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x = tf.layers.conv3d(
                x, filters, [3, 3,3], padding='SAME', strides=(stride, stride,stride))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def deconv(self, x, filters, stride=2):
        self.deconv_counter += 1

        with tf.variable_scope('deconv_unit_%d' % self.deconv_counter, reuse=self.reuse):
            print('deconv_unit_%d' % self.deconv_counter)

            x = tf.layers.conv3d_transpose(
                x, filters, [3, 3,3], padding='SAME', strides=(2, 2,2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x




    def n_convs(self, x, n, filters):
        dil = 1
        for _ in range(n):
            # x = self.conv(x, filters)
            x = self.xception_block(x, filters, kernel=3, last_stride=1, dilatation=dil, do_residual=True)
            dil+=1

        return x

    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):


            x = self.imgs


            x.set_shape([None, self.input_shape[0], self.input_shape[1],self.input_shape[2], self.input_channels])



            x = self.n_convs(x, 2, self.filters)

            x_link1 = x

            x = self.conv(x, 2 * self.filters, stride=2)
            x = self.n_convs(x, 2, 2 * self.filters)
            x_link2 = x

            x = self.conv(x, 4 * self.filters, stride=2)
            x = self.n_convs(x, 3, 4 * self.filters)
            x_link3 = x

            x = self.conv(x, 8 * self.filters, stride=2)
            x = self.n_convs(x, 3, 8 * self.filters)
            x_link4 = x

            x = self.conv(x, 8 * self.filters, stride=2)
            x = self.n_convs(x, 3, 8 * self.filters)

            x = self.deconv(x, 8 * self.filters)
            x = tf.concat([x, x_link4], 4)
            x = self.n_convs(x, 3, 8 * self.filters)

            x = self.deconv(x, 4 * self.filters)
            x = tf.concat([x, x_link3], 4)
            x = self.n_convs(x, 3, 4 * self.filters)

            x = self.deconv(x, 2 * self.filters)
            x = tf.concat([x, x_link2], 4)
            x = self.n_convs(x, 2, 2 * self.filters)

            x = self.deconv(x, self.filters)
            x = tf.concat([x, x_link1], 4)
            x = self.n_convs(x, 2, self.filters)
            x = tf.layers.conv3d(x, self.output_channels, [1, 1,1], padding='SAME')
            return x

