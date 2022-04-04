import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from model import YOLOv5
import numpy as np
import cv2
from data.dataset import Dataset
from tqdm import tqdm
from losses import YOLOLoss
from utils import draw_outputs
from config import Configuration
cfg = Configuration()


def transform_labels(y_true):
    # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
    true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
    true_class_idx = tf.one_hot(tf.cast(true_class_idx, dtype=tf.int32),cfg.num_classes, dtype=tf.float32)
    return true_box, true_obj, true_class_idx


def yolo_nms(outputs, classes):
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
    classes = tf.argmax(dscores,1)
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=0.1,
        soft_nms_sigma=0.5
    )
    
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


def draw_gt_labels(dataset, num_batches=5):
    pbar = tqdm(dataset.train_ds.take(num_batches))
    idx = 0
    for data_batches in pbar:
        for i in range(data_batches[0].shape[0]):
            idx += 1
            img, labels = data_batches[0][i],data_batches[1]
            output_list = []
            for j in range(cfg.yolo_anchor_masks.shape[0]):
                true_labels = transform_labels(labels[j][i])
                output_list.append(true_labels)
            boxes, scores, classes, nums = yolo_nms(output_list, 10)
            img = img.numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for k in range(nums[0]):
                img = draw_outputs(img, (boxes, scores, classes, nums), cfg.class_names)
            cv2.imwrite(os.path.join(out_path,f'{idx}.jpg'), img)
    return


# The gt labels are drawn after resize, letter-box padding and nms to mimic the actual detection pipeline
# So, there might be slight variations compared to the actual GT labels due to resizing and nms

out_path = 'preds'
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
dataset = Dataset()
draw_gt_labels(dataset, num_batches = 10)