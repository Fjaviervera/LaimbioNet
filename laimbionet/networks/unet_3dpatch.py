import tensorflow as tf


class Unet:

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


        self.conv_counter = 0
        self.deconv_counter = 0
        self.momentum = 0.99
        self.renorm = renorm
        self.output = self.build_net()

    def __str__(self):
        return ('U-net estandar designed by javier.vera@urjc.es')


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

        with tf.variable_scope('deconv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('deconv_unit_%d' % self.conv_counter)

            x = tf.layers.conv3d_transpose(
                x, filters, [3, 3,3], padding='SAME', strides=(2, 2,2))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')

            x = tf.nn.relu(x, name='relu')
            return x

    def n_convs(self, x, n, filters):
        for _ in range(n):
            x = self.conv(x, filters)
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

