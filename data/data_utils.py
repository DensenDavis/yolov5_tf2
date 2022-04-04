import tensorflow as tf

''' Converts boxes from center_x,center_y,width,height to left_x,top_y,right_x,bottom_y'''
@tf.function(experimental_relax_shapes=True) # Optional does'nt make much of a difference
def convert_to_corners(boxes):
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

''' Converts boxes from left_x,top_y,right_x,bottom_y to top_y,left_x,bottom_y,right_x'''
@tf.function(experimental_relax_shapes=True)
def swap_xy(boxes):
    return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

''' Converts boxes from left_x,top_y,right_x,bottom_y to center_x,center_y,width,height'''
@tf.function(experimental_relax_shapes=True)
def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )

def compute_iou(boxes1_corners, boxes2_corners):
    '''Take the 2 bounding boxes and returns their intersection area and IOU scores ex: (N,4),(M,4)-->(N,M)-Intersection area ratio,(N,M)-IOU'''
    boxes1 = convert_to_xywh(boxes1_corners)
    boxes2 = convert_to_xywh(boxes2_corners)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area/boxes2_area,0.0,1.0),tf.clip_by_value(intersection_area / union_area, 0.0,1.0)

def zero_pad_bboxes(images,labels, max_count = 100):
    bboxes, class_ids = labels
    box_mask = tf.zeros((max_count,4), tf.float32)
    cls_id_mask = tf.zeros((max_count), tf.float32)-1.0
    new_boxes = tf.concat([bboxes,box_mask], axis=0)[:max_count]
    new_labels = tf.concat([class_ids,cls_id_mask], axis=0)[:max_count]
    labels = tf.concat([new_boxes,new_labels], axis=-1)
    return images, labels

def zero_pad_bboxes_batch(images,labels, max_count = 100):
    bboxes, class_ids = labels
    box_mask = tf.zeros((tf.shape(images)[0],max_count,4), tf.float32)
    cls_id_mask = tf.zeros((tf.shape(images)[0],max_count), tf.float32)-1.0
    new_boxes = tf.concat([bboxes,box_mask], axis=1)[:,:max_count]
    new_labels = tf.concat([class_ids,cls_id_mask], axis=1)[:,:max_count]
    labels = tf.concat([new_boxes,tf.expand_dims(new_labels,axis=-1)], axis=-1)
    return images, labels