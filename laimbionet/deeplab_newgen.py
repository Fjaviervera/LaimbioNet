import tensorflow as tf
slim = tf.contrib.slim
import math




class unet_xception:
    def __init__(self, name, input, input_channels, is_training=True, reuse=None, filters=16, output_stride = 16,
                 batch_size=None):
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
        self.output_stride = output_stride
        self.output_stride_track = 1
        self.filters_multiplier = 1
        self.dilatation_track = 2
        self.deconv_counter = 0
        self.output_channels = 1
        self.output = self.build_net()


    def xception_block(self,inputs, filters, kernel, last_stride = 1, dilatation=1, do_residual = True):

        self.xception_block_counter += 1

        with tf.variable_scope('Xception_block_counter_%d' % self.xception_block_counter, reuse=self.reuse):
            print('Xception block counter %d' % self.xception_block_counter)



            if do_residual:

                if last_stride!=1 or int(inputs.shape[3])!=filters:
                    x_res = tf.nn.relu(inputs, name='relu_input_residual')

                    x_res = tf.layers.conv3d(
                        x_res, filters, 1, padding='SAME', strides=last_stride,dilation_rate = 1)

                    x_res = tf.layers.batch_normalization(x_res, training=self.is_training, reuse=self.reuse,
                                                      momentum=self.momentum, renorm=self.renorm, name='bn_residual')
                else:
                    x_res = inputs

            else:
                x_res = inputs

            x = tf.nn.relu(inputs, name='relu_input')
            x = tf.layers.conv3d(
                x, filters, kernel, padding='SAME', strides=1, dilation_rate=1)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                                  momentum=self.momentum, renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x, name='relu_1')
            x = tf.layers.conv3d(
                x, filters, kernel, padding='SAME', strides=1, dilation_rate=1)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')

            x = tf.nn.relu(x, name='relu_2')



            x = tf.layers.conv3d(
                x, filters, kernel, padding='SAME', strides=last_stride, dilation_rate=dilatation)

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')

            if do_residual:
                x = x + x_res

            return x



    def xception_flow(self,x, filters, do_last_stride = False, do_residual = True):

        if do_last_stride:
            if self.output_stride_track < self.output_stride:
                x = self.xception_block(x, filters, kernel=3, last_stride=True, dilatation=1,
                                        do_residual=do_residual)
                print('aplico stride %d' % self.output_stride_track)
            else:
                x = self.xception_block(x, filters, kernel=3, last_stride=True, dilatation=self.dilatation_track,
                                        do_residual=do_residual)

                print('aplico dilatation de %d' % self.dilatation_track)
                self.dilatation_track *= 2

        x = self.xception_block(x, filters, kernel=3, last_stride=False, dilatation=1, do_residual=do_residual)
        x = self.xception_block(x, filters, kernel=3, last_stride=False, dilatation=1, do_residual=do_residual)


        return x

    def deconv(self, x, filters):
        if self.dilatation_track > 2:
            self.deconv_counter += 1



            self.dilatation_track /= 2

            with tf.variable_scope('deconv_unit_%d' % self.conv_counter, reuse=self.reuse):
                print('deconv_unit_%d' % self.conv_counter)

                x = tf.layers.conv3d_transpose(
                    x, filters, [3, 3,3], padding='SAME', strides=(2, 2,2))
                x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                                  momentum=self.momentum, renorm=self.renorm, name='bn')


                x = tf.nn.relu(x, name='relu')
                return x
        else:
            print('skipped deconv')
            return x



    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):

            x = self.imgs


            x.set_shape([None, 32, 32,32, self.input_channels])

            x = self.xception_flow(x, self.filters, do_last_stride=False, do_residual=True)
            link1 = x
            x = self.xception_flow(x,self.filters*2 , do_last_stride=True, do_residual=True)
            x = self.xception_flow(x, self.filters*2, do_last_stride=False, do_residual=True)
            link2 = x
            x = self.xception_flow(x, self.filters*4, do_last_stride=True, do_residual=True)
            x = self.xception_flow(x, self.filters*4, do_last_stride=False, do_residual=True)
            link3 = x
            x = self.xception_flow(x, self.filters*8, do_last_stride=True, do_residual=True)
            x = self.xception_flow(x, self.filters*8, do_last_stride=False, do_residual=True)
            link4 = x
            x = self.xception_flow(x, self.filters*16, do_last_stride=True, do_residual=True)
            x = self.xception_flow(x, self.filters*16, do_last_stride=False, do_residual=True)


            x = self.deconv( x, self.filters*8)

            x = tf.concat([x, link4], 4)
            x = self.xception_flow(x, self.filters * 8, do_last_stride=False, do_residual=True)

            x = self.deconv( x, self.filters*4)

            x = tf.concat([x, link3], 4)
            x = self.xception_flow(x, self.filters * 4, do_last_stride=False, do_residual=True)

            x = self.deconv( x, self.filters*2)
            x = tf.concat([x, link2], 4)

            x = self.xception_flow(x, self.filters * 2, do_last_stride=False, do_residual=True)
            x = self.deconv( x, self.filters)
            x = tf.concat([x, link1], 4)
            x = self.xception_flow(x, self.filters , do_last_stride=False, do_residual=True)
            x = tf.layers.conv3d(x, self.output_channels, [1, 1, 1], padding='SAME')





            return x
