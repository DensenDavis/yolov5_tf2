import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy,sparse_categorical_crossentropy
from config import Configuration
cfg = Configuration()

class YOLOLoss(tf.losses.Loss):
    def __init__(self, anchors):
        super(YOLOLoss, self).__init__(reduction="none", name="YOLOLoss")
        self.anchors = tf.constant(anchors)

    def _meshgrid(self, n_a, n_b):

        return [
            tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
            tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
        ]

    def broadcast_iou(self, box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))

        # broadcast boxes
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                        tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                        tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
            (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
            (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    def yolo_boxes(self, pred, classes):
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_xywh = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = self._meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * self.anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        pred_box = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return pred_box, objectness, class_probs, pred_xywh

    def call(self, y_true, y_pred):
        # 1. transform all pred outputs
        # y_true: (N, grid, grid, anchors, [x1, y1, x2, y2, , cls_0, cls_1, ...])
        # y_pred: (N, grid, grid, anchors, [x, y, w, h, obj, cls_0, cls_1, ...])
        pred_box, pred_obj, pred_class, pred_xywh = self.yolo_boxes(y_pred, cfg.num_classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (N, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / self.anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(self.broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < cfg.train_iou_threshold, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
