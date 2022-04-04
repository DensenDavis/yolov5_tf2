import tensorflow as tf
from tensorflow.keras import layers

class Focus(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='SAME',**kwargs):
        super(Focus, self).__init__(name="Focus", **kwargs)
        self.conv = Conv(filters, kernel_size, strides)

    def call(self, x):
        return self.conv(tf.nn.space_to_depth(x, 2))


class Conv(layers.Layer):
    def __init__(self,filters, kernel_size, strides,**kwargs):
        super(Conv, self).__init__(name="Conv", **kwargs)
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__strides = strides
        self.bn = layers.BatchNormalization(momentum=0.03)
        self.activation = layers.Activation(tf.keras.activations.swish)

    def build(self, input_shape):
        if self.__strides == 2:
            self.conv = tf.keras.Sequential([layers.Input(input_shape[1:])])
            self.conv.add(layers.ZeroPadding2D(((1, 0), (1, 0))))
            self.conv.add(layers.Conv2D(self.__filters, self.__kernel_size, self.__strides, "valid", use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4)))
        else:
            self.conv = layers.Conv2D(self.__filters, self.__kernel_size, self.__strides, "same", use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4))

    def call(self, x):
        return self.activation(self.bn(self.conv(x)))

class Bottleneck(layers.Layer):
    def __init__(self, filters, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(int(filters * expansion), 1, 1)
        self.conv2 = Conv(filters, 3, 1)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class BottleneckCSP(layers.Layer):
    def __init__(self, filters, n_layer=1, shortcut=True, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        filters_e = int(filters * expansion)
        self.conv1 = Conv(filters_e, 1, 1)
        self.conv2 = layers.Conv2D(filters_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = layers.Conv2D(filters_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv4 = Conv(filters, 1, 1)
        self.bn = layers.BatchNormalization(momentum=0.03)
        self.activation = layers.Activation(tf.keras.activations.swish)
        self.modules = tf.keras.Sequential([Bottleneck(filters_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        y1 = self.conv3(self.modules(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.activation(self.bn(tf.concat([y1, y2], axis=-1))))

class SPP(layers.Layer):
    #  YOLOv3 Spatial pyramid pooling
    def __init__(self, filters, k_sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        self.conv2 = Conv(filters, 1, 1)
        self.max_pools = [layers.MaxPool2D(pool_size=k, strides=1, padding='same') for k in k_sizes]

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv1 = Conv(in_channels//2, 1, 1)

    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(tf.concat([x] + [pool(x) for pool in self.max_pools], axis=-1))