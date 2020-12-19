import tensorflow as tf
slim = tf.contrib.slim
import math


def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.
  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs










class Network_DeepLab_v3_xception:
    def __init__(self, name, input, input_channels, is_training=True, reuse=None, filters=16, padding = 'VALID',output_stride = 8,middle_blocks=3, aspp_blocks = 1,
                 batch_size=None,  dropout=False, dropout_prob=0.5,
                 renorm=False):
        self.imgs = input
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.filters = filters
        self.padding = padding
        self.separable_conv_counter = 0
        self.xception_block_counter = 0
        self.middle_blocks = middle_blocks
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.aspp_blocks = aspp_blocks
        self.entry_flow_counter = 0
        self.middle_flow_counter = 0
        self.atrous_counter = 0
        self.aspp_counter = 0
        self.conv_counter = 0
        self.momentum = 0.99
        self.renorm = renorm
        self.output_stride = output_stride
        self.output_stride_track = 1
        self.filters_multiplier = 1
        self.dilatation_track = 2
        self.output = self.build_net()




    def xception_block(self,inputs, filters, kernel, stride = (1,1,2), dilatation=1, do_residual = True):

        self.xception_block_counter += 1

        with tf.variable_scope('Xception_block_counter_%d' % self.xception_block_counter, reuse=self.reuse):
            print('Xception block counter %d' % self.xception_block_counter)

            if isinstance(filters, list):
                if len(filters) != 3:
                    raise ValueError(' If list expects three elements in filters.')

            else:
                filters =[filters,filters,filters]




            if do_residual:

                if stride[2]!=1 or int(inputs.shape[3])!=filters[2]:
                    x_res = tf.nn.relu(inputs, name='relu_input_residual')

                    x_res = tf.layers.conv3d(
                        x_res, filters[2], 1, padding='SAME', strides=stride[2],dilation_rate = dilatation)

                    x_res = tf.layers.batch_normalization(x_res, training=self.is_training, reuse=self.reuse,
                                                      momentum=self.momentum, renorm=self.renorm, name='bn_residual')
                else:
                    x_res = inputs

            else:
                x_res = inputs

            x = tf.nn.relu(inputs, name='relu_input')
            x = tf.layers.conv3d(
                x, filters[0], kernel, padding='SAME', strides=stride[0], dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                                  momentum=self.momentum, renorm=self.renorm, name='bn1')

            x = tf.nn.relu(x, name='relu_1')
            x = tf.layers.conv3d(
                x, filters[1], kernel, padding='SAME', strides=stride[1], dilation_rate=dilatation)
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn2')

            x = tf.nn.relu(x, name='relu_2')
            x = tf.layers.conv3d(
                x, filters[2], kernel, padding='SAME', strides=stride[2], dilation_rate=dilatation)

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn3')

            if do_residual:
                x = x + x_res

            return x



    def xception_flow(self,x, filters_list, strides =(1,1,1), do_residual = True):



        for filters in filters_list:

            if 2 in strides:

                if self.output_stride_track < self.output_stride :

                    x = self.xception_block(x, filters, kernel = 3, stride = strides, dilatation=1, do_residual = do_residual)


                    self.output_stride_track *= 2
                    print('aplico stride %d' %self.output_stride_track)
                else:

                    x = self.xception_block(x, filters, kernel=3, stride=(1,1,1), dilatation = self.dilatation_track, do_residual = do_residual)

                    print('aplico dilatation de %d' % self.dilatation_track)

                    self.dilatation_track *= 2

            else:

                x = self.xception_block(x, filters, kernel=3, stride=strides, dilatation=1, do_residual=do_residual)

        return x

    def aspp(self, x, filters):
        self.aspp_counter += 1
        with tf.variable_scope('aspp_unit_%d' % self.aspp_counter, reuse=self.reuse):
            print('aspp_unit_%d' % self.aspp_counter)

            x = tf.nn.relu(x, name='relu')

            x_1 = tf.layers.conv3d(
                x, filters, 3, padding='SAME', strides=1, dilation_rate=2)


            x_1 = tf.layers.batch_normalization(x_1, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn1')
            x_1 = tf.nn.relu(x_1, name='relu1')
            x_2 = tf.layers.conv3d(
                x, filters, 3, padding='SAME', strides=1, dilation_rate=3)
            x_2 = tf.layers.batch_normalization(x_2, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn2')
            x_2 = tf.nn.relu(x_2, name='relu2')
            x_3 = tf.layers.conv3d(
                x, filters, 3, padding='SAME', strides=1, dilation_rate=4)
            x_3 = tf.layers.batch_normalization(x_3, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn3')
            x_3 = tf.nn.relu(x_3, name='relu3')
            x_4 = tf.layers.conv3d(
                x, filters, 1, padding='SAME', strides=1, dilation_rate=1)
            x_4 = tf.layers.batch_normalization(x_4, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn4')
            x_4 = tf.nn.relu(x_4, name='relu4')
            x_5 = tf.layers.average_pooling3d(x, [2, 2,2], [1, 1,1], padding='SAME')


            x = tf.concat([x_1, x_2, x_3, x_4, x_5], 4)

            x = tf.layers.conv3d(
                x, filters, 1, padding='SAME', strides=1, dilation_rate=1)

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=self.momentum,
                                              renorm=self.renorm, name='bn')

            return x


    def decoder_flow(self, x1, x2, x3, filters):

        with tf.variable_scope('Decoder_flow', reuse=self.reuse):
            print('Decoder_flow')

            x1 = tf.nn.relu(x1, name='relu1')
            x2 = tf.nn.relu(x2, name='relu2')
            x3 = tf.nn.relu(x3, name='relu3')

            x1 = self.xception_block(x1, 64, kernel = 3, stride=(1, 1, 1))
            x2 = self.xception_block(x2, 64, kernel =3, stride=(1, 1, 1))

            x = tf.concat([x2, x3], 4)



            for up_iter in range(int(math.log(self.output_stride,2.0) )):

                print('up iter in decoder %d'%up_iter)
                if x.shape[2] == x1.shape[2]:
                    print('concat inputs in decoder')
                    x = tf.concat([x, x1], 4)


                x = tf.layers.conv3d_transpose(
                    x, filters, [3, 3, 3], padding='SAME', strides=(2, 2,2))
                x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                                  momentum=self.momentum, renorm=self.renorm, name='bn_' + str(up_iter))
                x = tf.nn.relu(x, name='relu_' + str(up_iter))

                x = self.xception_block(x,filters, 3, stride=(1,1,1))

                x = tf.nn.relu(x, name='relu2_' + str(up_iter))



                if x.shape[2] == x1.shape[2]:
                    print('concat inputs in decoder')
                    x = tf.concat([x, x1], 4)

            x = tf.layers.conv3d(x, 1, 1, padding='SAME')

            return x




    def conv_init(self, x, filters):

        with tf.variable_scope('conv_init', reuse=self.reuse):
            print('conv_init')


            x = tf.layers.conv3d(
                x, filters, [3, 3, 3], padding='SAME', strides=(1, 1,1))



            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse,
                                              momentum=self.momentum, renorm=self.renorm, name='bn1')

            # x = tf.nn.relu(x, name='relu1')





        return x

    def build_net(self):
        print('Building Network:')
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):

            x = self.imgs


            x.set_shape([None, 32, 32,32, self.input_channels])

            x = self.conv_init(x, self.filters)

            x = self.xception_flow(x, [self.filters*2], strides = [1,1,1])
            x = self.xception_flow(x,  [self.filters*2], strides=[1, 1, 1])

            x = self.xception_flow(x,  [self.filters*4], strides=[1, 1, 1])
            x = self.xception_flow(x, [self.filters *4], strides=[1, 1, 1])
            x = self.aspp( x, 128)

            x = tf.nn.relu(x, name='relu_final')
            x = self.xception_flow(x, [self.filters * 4], strides=[1, 1, 1])
            x = tf.nn.relu(x, name='relu_final2')
            x = tf.layers.conv3d(x, 1, 1, padding='SAME')


            return x
