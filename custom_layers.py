import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization,MaxPool2D, ZeroPadding2D

class Focus(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='SAME'):
        super(Focus, self).__init__(name="Focus", **kwargs)
        self.conv = Conv(filters, kernel_size, strides, padding)

    def call(self, x):
        return self.conv(tf.nn.space_to_depth(x, 2))


class Conv(Layer):
    def __init__(self,filters, kernel_size, strides):
        super(Conv, self).__init__(name="Conv", **kwargs)
        self.__strides = strides
        padding = "valid" if strides == 2 else "same"
        self.conv = Conv2D(filters, kernel_size, strides, padding, use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4))
        self.bn = BatchNormalization(momentum=0.03)
        self.activation = layers.Activation(tf.keras.activations.swish)

    def build(self, input_shape):
        if self.__strides == 2:
            self.conv = tf.keras.Sequential(
                [
                    layers.Input([None,None,tf.shape(input_shape)[-1]]),
                    layers.ZeroPadding2D(((1, 0), (1, 0)))
                ])
        super().build(input_shape)

    def call(self, x):
        return self.activation(self.bn(self.conv(x)))

class Bottleneck(Layer):
    def __init__(self, filters, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(int(filters * expansion), 1, 1)
        self.conv2 = Conv(filters, 3, 1)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class BottleneckCSP(Layer):
    def __init__(self, filters, n_layer=1, shortcut=True, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        filters_e = int(filters * expansion)
        self.conv1 = Conv(filters_e, 1, 1)
        self.conv2 = Conv2D(filters_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv2D(filters_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv4 = Conv(units, 1, 1)
        self.bn = BatchNormalization(momentum=0.03)
        self.activation = layers.Activation(tf.keras.activations.swish)
        self.modules = tf.keras.Sequential([Bottleneck(filters_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        y1 = self.conv3(self.modules(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.activation(self.bn(tf.concat([y1, y2], axis=-1))))

class SPP(Layer):
    #  YOLOv3 Spatial pyramid pooling
    def __init__(self, filters, k_sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        self.conv2 = Conv(filters, 1, 1)
        self.max_pools = [MaxPool2D(pool_size=k, strides=1, padding='same') for k in k_sizes]

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv1 = Conv(in_channels//2, 1, 1)
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(tf.concat([x] + [pool(x) for pool in self.max_pools], axis=-1))