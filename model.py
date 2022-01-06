import tensorflow as tf
from tensorflow.keras import layers
from custom_layers import Focus, Conv, BottleneckCSP, SPP
from config import Configuration
cfg = Configuration()

class YOLOv5Backbone(tf.keras.Model):
    def __init__(self, depth, width, **kwargs):
        super(YOLOv5Backbone, self).__init__(name="Head", **kwargs)
        self.focus = Focus(int(round(width * 64)),3)
        self.conv1 = Conv(int(round(width * 128)), 3, 2)
        self.conv2 = Conv(int(round(width * 256)), 3, 2)
        self.conv3 = Conv(int(round(width * 512)), 3, 2)
        self.conv4 = Conv(int(round(width * 1024)), 3, 2)
        self.csp1 = BottleneckCSP(int(round(width * 128)), int(round(depth * 3)))
        self.csp2 = BottleneckCSP(int(round(width * 256)), int(round(depth * 9)))
        self.csp3 = BottleneckCSP(int(round(width * 512)), int(round(depth * 9)))
        self.csp4 = BottleneckCSP(int(round(width * 1024)), int(round(depth * 3)), False)
        self.spp = SPP(width * 1024)
    
    def call(self, inputs):
        x = self.focus(inputs)
        x = self.conv1(x)
        x = self.csp1(x)
        x = self.conv2(x)
        out_1 = x = self.csp2(x)
        x = self.conv3(x)
        out_2 = x = self.csp3(x)
        x = self.conv4(x)
        x = self.spp(x)
        out_3 = x = self.csp4(x)
        return [out_1,out_2,out_3]

class YOLOv5Head(tf.keras.Model):
    def __init__(self, depth, width, **kwargs):
        super(YOLOv5Neck, self).__init__(name="Neck", **kwargs)
        self.conv1 = Conv(int(round(width * 512)), 1, 1)
        self.conv2 = Conv(int(round(width * 256)), 1, 1)
        self.conv3 = Conv(int(round(width * 256)), 3, 2)
        self.conv4 = Conv(int(round(width * 512)), 3, 2)

        self.upsample1 = layers.UpSampling2D()
        self.upsample2 = layers.UpSampling2D()

        self.concat1 = layers.Concatenate(axis=-1)
        self.concat2 = layers.Concatenate(axis=-1)
        self.concat3 = layers.Concatenate(axis=-1)
        self.concat4 = layers.Concatenate(axis=-1)

        self.csp1 = BottleneckCSP(int(round(width * 512)), int(round(depth * 3)), False)
        self.csp2 = BottleneckCSP(int(round(width * 256)), int(round(depth * 3)), False)
        self.csp3 = BottleneckCSP(int(round(width * 512)), int(round(depth * 3)), False)
        self.csp4 = BottleneckCSP(int(round(width * 1024)), int(round(depth * 3)), False)

    def call(self, inputs):
        bbf_1, bbf_2, bbf_3 = inputs # backbone features

        x1 = x = self.conv1(bbf_3)
        x = self.upsample1(x)
        x = self.concat1(x,bbf_2)
        x = self.csp1(x)

        x2 = x = self.conv2(x)
        x = self.upsample2(x)
        x = self.concat2(x,bbf_1)
        out_1 = x = self.csp2(x)

        x = self.conv3(x)
        x = self.concat3(x,x2)
        out_2 = x = self.csp3(x)

        x = self.conv4(x)
        x = self.concat4(x,x1)
        out_3 = x = self.csp4(x)
        return [out_1, out_2, out_3]

class YOLOv5(tf.keras.Model):
    def __init__(self, **kwargs):
        super(YOLOv5, self).__init__(name="YOLOv5", **kwargs)
        self.backbone = YOLOv5Backbone(depth, width)
        self.head = YOLOv5Head(depth, width)
        self.convs = [layers.Conv2D(cfg.num_anchors * (cfg.num_classes + 5), 1, name=f'out_{1}',
                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.keras.regularizers.L2(5e-4)) for i in cfg.num_strides]
        self.feature_reshape_layers = [Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            cfg.num_anchors, cfg.num_classes + 5))) for i in range(cfg.num_strides)]
        self.box_processing = [Lambda(lambda x: yolo_boxes(x, cfg.anchors[cfg.anchor_masks[i]], cfg.num_classes),
                     name=f'yolo_boxes_{i}') for i in range(cfg.num_strides)]

    def call(self, image, training=False):
        backbone_features = self.backbone(image, training=training)
        out_features = self.head(backbone_features)
        for i in range(cfg.num_strides):
            out_features[i] = self.convs[i](out_features[i])
            out_features[i] = self.feature_reshape_layers[i](out_features[i])
        return out_features