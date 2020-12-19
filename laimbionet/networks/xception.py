import tensorflow as tf
slim = tf.contrib.slim
import math




class xception:
    def __init__(self, name, input, input_channels, is_training=True, reuse=None, filters=64,
                 batch_size=None, xceptions_rounds = 9):
        self.imgs = input
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.filters = filters
        self.separable_conv_counter = 0
        self.xception_block_counter = 0
        self.entry_flow_counter = 0
        self.middle_flow_counter = 0
        self.atrous_counter = 0
        self.aspp_counter = 0
        self.conv_counter = 0
        self.momentum = 0.99
        self.xception_rounds = xceptions_rounds
        self.output_stride_track = 1
        self.filters_multiplier = 1
        self.dilatation_track = 2
        self.deconv_counter = 0
        self.output_channels = 1
        self.renorm = False
        self.output = self.build_net()


    def xception_block(self,inputs, filters, kernel, last_stride = 1, dilatation=1, do_residual = True):

        self.xception_block_counter += 1

        with tf.variable_scope('Xception_block_counter_%d' % self.xception_block_counter, reuse=self.reuse):
            print('Xception block counter %d' % self.xception_block_counter)



            if do_residual:

                if last_stride!=1 or int(inputs.shape[3])!=filters:
                    x_res = tf.nn.relu(inputs, name='relu_input_residual')

                    x_res = tf.layers.conv2d(
                        x_res, filters, 1, padding='SAME', strides=last_stride,dilation_rate = 1)

                    x_res = tf.layers.batch_normalization(x_res, training=self.is_training, reuse=self.reuse,
                                                      momentum=self.momentum, renorm=self.renorm, name='bn_residual')
                else:
                    x_res = inputs

            else:
                x_res = inputs

            x = tf.nn.relu(inputs, name='relu_input')
            x = tf.layers.conv2d(
                x, filters, kernel, padding='SAME', strides=1, dilation_rate=1)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                                  momentum=self.momentum, renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x, name='relu_1')
            x = tf.layers.conv2d(
                x, filters, kernel, padding='SAME', strides=1, dilation_rate=1)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')

            x = tf.nn.relu(x, name='relu_2')



            x = tf.layers.conv2d(
                x, filters, kernel, padding='SAME', strides=last_stride, dilation_rate=dilatation)

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')

            if do_residual:
                x = x + x_res

            return x


    def deconv(self, x, filters):


        with tf.variable_scope('deconv_unit_%d' % self.deconv_counter, reuse=self.reuse):
            print('deconv_unit_%d' % self.deconv_counter)
            self.deconv_counter+=1
            x = tf.layers.conv2d_transpose(
                x, filters, [3, 3], padding='SAME', strides=(2, 2), reuse=self.reuse)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')


            x = tf.nn.relu(x, name='relu')
            return x

    def conv(self, x, filters, stride=1, kernel =3, last_relu = True):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % self.conv_counter, reuse=self.reuse):
            print('conv_unit_%d' % self.conv_counter)

            x = tf.layers.conv2d(
                x, filters, [kernel, kernel], padding='SAME', strides=(stride, stride))
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn')
            if last_relu:
                 return tf.nn.relu(x, name='relu')
            return x



    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):

            x = self.imgs


            x.set_shape([None, None, None, self.input_channels])

            x = self.conv(x, self.filters, kernel = 7)
            x = self.conv(x, 2*self.filters, kernel=3, stride=2)
            x = self.conv(x, 4 * self.filters, kernel=3, stride=2, last_relu=False)

            for _ in range(self.xception_rounds):
                x = self.xception_block(x,4*self.filters,kernel = 3)

            x = self.deconv(x,2*self.filters)
            x = self.deconv(x,  self.filters)
            x = tf.layers.conv2d(
                x, 1, [7, 7], padding='SAME')


            return x
