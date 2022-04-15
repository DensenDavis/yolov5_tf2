import tensorflow as tf
import numpy as np
import os.path as osp
import cv2
import os
from tqdm import tqdm
from utils import draw_outputs
from losses import YOLOLoss
from config import Configuration
cfg = Configuration()

class TrainLoop():
    def __init__(self, dataset, model, optimizer):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.stride_losses = [YOLOLoss(cfg.yolo_anchors[mask]) for mask in cfg.yolo_anchor_masks]
        self.mean_train_loss = tf.keras.metrics.Mean(name='mean_train_loss')
        self.mean_val_loss = tf.keras.metrics.Mean(name='val_loss')

    @tf.function
    def calculate_loss(self, y_true, y_pred):
        net_loss = 0.0
        for i in range(cfg.num_strides):
            net_loss += self.stride_losses[i](y_true[i],y_pred[cfg.num_strides-1-i])
        return net_loss

    @tf.function
    def train_step(self, input_batch, gt_batch):
        with tf.GradientTape(persistent=False) as tape:
            output_batch = self.model(input_batch, training=True)
            net_loss = self.calculate_loss(gt_batch, output_batch)
        gradients = tape.gradient(net_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        self.mean_train_loss(net_loss)
        return

    def train_one_epoch(self, epoch):
        self.mean_train_loss.reset_states()
        pbar = tqdm(self.dataset.train_ds, desc=f'Epoch : {epoch.numpy()}')
        for data_batch in pbar:
            input_batch, gt_batch = data_batch
            self.train_step(input_batch, gt_batch)
        return

    # @tf.function
    def val_step(self, input_batch, gt_batch):
        output_batch = self.model(input_batch, training=False)
        net_loss = self.calculate_loss(gt_batch, output_batch)
        self.mean_val_loss(net_loss)
        return output_batch

    def yolo_boxes(self, pred, anchors):
        def _meshgrid(n_a, n_b):
            return [
                tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
                tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
            ]
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, cfg.num_classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box

    def yolo_nms(self, outputs, classes):
        b, c, t = [], [], []

        for o in outputs:
            o = list(o)
            o[0] = tf.expand_dims(o[0], axis=0)
            o[1] = tf.expand_dims(o[1], axis=0)
            o[2] = tf.expand_dims(o[2], axis=0)
            b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

        bbox = tf.concat(b, axis=1)
        confidence = tf.concat(c, axis=1)
        class_probs = tf.concat(t, axis=1)

        # If we only have one class, do not multiply by class_prob (always 0.5)
        if classes == 1:
            scores = confidence
        else:
            scores = confidence * class_probs

        dscores = tf.squeeze(scores, axis=0)
        scores = tf.reduce_max(dscores,[1])
        bbox = tf.reshape(bbox,(-1,4))
        bbox = tf.stack([bbox[...,1],bbox[...,0],bbox[...,3],bbox[...,2]], axis=1) # [N,[y1,x1,y2,x2]] format
        classes = tf.argmax(dscores,1)
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            boxes=bbox,
            scores=scores,
            max_output_size=1000,
            iou_threshold=0.1,
            score_threshold=0.3,
            soft_nms_sigma=0.5
        )
        bbox = tf.stack([bbox[...,1],bbox[...,0],bbox[...,3],bbox[...,2]], axis=1) # [N,[x1,y1,x2,y2]] format

        num_valid_nms_boxes = tf.shape(selected_indices)[0]

        selected_indices = tf.concat([selected_indices,tf.zeros(1000-num_valid_nms_boxes, tf.int32)], 0)
        selected_scores = tf.concat([selected_scores,tf.zeros(1000-num_valid_nms_boxes,tf.float32)], -1)

        boxes=tf.gather(bbox, selected_indices)
        boxes = tf.expand_dims(boxes, axis=0)
        scores=selected_scores
        scores = tf.expand_dims(scores, axis=0)
        classes = tf.gather(classes,selected_indices)
        classes = tf.expand_dims(classes, axis=0)
        valid_detections=num_valid_nms_boxes
        valid_detections = tf.expand_dims(valid_detections, axis=0)
        return boxes, scores, classes, valid_detections


    def save_predictions(self, input_batch, output_batch, epoch, sample_count):
        output_list = []
        for j in range(cfg.num_strides):
            pred_labels = self.yolo_boxes(output_batch[cfg.num_strides-1-j], cfg.yolo_anchors[cfg.yolo_anchor_masks[j]])
            output_list.append(pred_labels)

        # boxes, scores, classes, nums = self.yolo_nms(output_list, cfg.num_classes)
        for i in range(input_batch.shape[0]):
            sample_count += 1
            if sample_count>cfg.display_samples:
                return sample_count
            img = input_batch[i].numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            tmp_list = []
            for stride in range(cfg.num_strides):
                tmp_list.append([output_list[stride][0][i],output_list[stride][1][i],output_list[stride][2][i],output_list[stride][3][i]])

            boxes, scores, classes, nums = self.yolo_nms(tmp_list, cfg.num_classes)
            # if i>=boxes.shape[0]:continue
            for k in range(nums[0]):
                img = draw_outputs(img, (boxes, scores, classes, nums), cfg.class_names)
            cv2.imwrite(os.path.join('val_img_log',f'{epoch}','pred',f'{sample_count}.jpg'), img)
        return sample_count

    def run_validation(self, save_prediction, epoch):
        '''
        If needed calculate validation loss as well.
        This may result in slower training due to extra computations.
        '''
        # self.val_loss.reset_states()
        self.mean_val_loss.reset_states()
        sample_count = 0
        for i, data_batch in enumerate(self.dataset.val_ds, start=1):
            input_batch, gt_batch = data_batch
            output_batch = self.val_step(input_batch, gt_batch)
            if(save_prediction):
                os.makedirs(osp.join('val_img_log',f'{epoch}','pred'), exist_ok = True)
                os.makedirs(osp.join('val_img_log',f'{epoch}','gt'), exist_ok = True)
                sample_count = self.save_predictions(input_batch, output_batch, epoch, sample_count)
                # self.save_gt(input_batch, gt_batch)
                if(sample_count >= cfg.display_samples):
                    save_prediction = False
        return
