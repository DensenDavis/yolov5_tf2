import tensorflow as tf
from config import Configuration
cfg = Configuration()
from data.augment import random_crop

def decode_labels(labels):
    # labels : string Tensor : [[cls_id x y w h]], xywh ∈ [0,1]
    # returns tuple (bboxes : xywh, cls_id)
    labels = tf.strings.strip(labels)
    labels = tf.strings.split(labels,'\n')
    labels = tf.strings.split(labels,' ')
    labels = tf.strings.to_number(labels)
    labels = labels.to_tensor()
    bbox = labels[:,1:]
    cls_id  = tf.cast(labels[:,0], dtype=tf.float32)
    return bbox,cls_id

def decode_images(image):
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.cast(image,tf.float32)/255.0
    return image

def pad_to_fixed_size(image,labels):
    image_shape =  tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    ratio = cfg.train_img_size/tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32),antialias=True)
    image = tf.image.pad_to_bounding_box(image, 0, 0, tf.cast(cfg.train_img_size,tf.int32),tf.cast(cfg.train_img_size,tf.int32))
    bbox,cls_id = labels[0],labels[1]
    bbox = tf.stack([(bbox[...,0]*image_shape[1])/cfg.train_img_size,(bbox[...,1]*image_shape[0])/cfg.train_img_size,\
        (bbox[...,2]*image_shape[1])/cfg.train_img_size,(bbox[...,3]*image_shape[0])/cfg.train_img_size],axis=-1)
    # bbox = bbox/cfg.train_img_size
    return image,(bbox,cls_id)

def resize_and_pad(img, labels):
    bboxes,cls_id  = labels
    img_dim = tf.shape(img)[:2]
    h,w = tf.cast(img_dim[0], tf.float32), tf.cast(img_dim[1], tf.float32)
    target_h = target_w = cfg.train_img_size
    scale = tf.minimum(target_h/h, target_w/w)

    bboxes_x = bboxes[:,0]*w
    bboxes_y = bboxes[:,1]*h
    bboxes_w = bboxes[:,2]*w
    bboxes_h = bboxes[:,3]*h
    bboxes = tf.stack([bboxes_x, bboxes_y, bboxes_w, bboxes_h],axis=-1)

    nh = tf.cast(scale * h, tf.int32)
    nh = tf.cond(nh%2==0,lambda:nh, lambda:nh+1)
    nw = tf.cast(scale * w, tf.int32)
    nw = tf.cond(nw%2==0,lambda:nw, lambda:nw+1)

    img_resized = tf.image.resize(img, [nh, nw], method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False,antialias=True)
    dh, dw = (target_h - nh) // 2, (target_w - nw) // 2

    x_pad = tf.cast(tf.fill([nh,dw,3],value=0), tf.float32)
    y_pad = tf.cast(tf.fill([dh,target_w,3],value=0), tf.float32)
    img_paded = tf.concat([x_pad, img_resized,x_pad], axis = 1)
    img_paded = tf.concat([y_pad, img_paded,y_pad], axis = 0)

    dw = tf.cast(dw, tf.float32)
    dh = tf.cast(dh, tf.float32)

    bboxes_x = (bboxes[:,0]*scale + dw)/target_w
    bboxes_y = (bboxes[:,1]*scale + dh)/target_h
    bboxes_w = (bboxes[:,2]*scale)/target_w
    bboxes_h = (bboxes[:,3]*scale)/target_h
    labels = tf.stack([bboxes_x, bboxes_y, bboxes_w, bboxes_h, cls_id],axis=-1)
    return img_paded,labels

def crop_or_pad_to_fixed_size(image,labels):
    image_shape =  tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    image,(bbox,cls_id) = tf.cond(tf.logical_or(image_shape[0]<=cfg.train_img_size,image_shape[1]<=cfg.train_img_size),\
        lambda:pad_to_fixed_size(image,labels),lambda:random_crop(image,labels))
    return image,(bbox,cls_id)

def bbox_rescale(image,labels):
    bbox,cls_id = labels
    postive_mask = tf.logical_or(tf.logical_or(bbox[:,0]!=0.0,bbox[:,1]!=0.0),tf.logical_or(bbox[:,2]!=0.0,bbox[:,3]!=0.0))
    postive_indices = tf.where(postive_mask)
    new_bboxes = tf.gather_nd(bbox,postive_indices)
    # new_bboxes = new_bboxes*cfg.train_img_size
    new_bboxes = tf.stack([new_bboxes[...,0]*cfg.train_img_size,new_bboxes[...,1]*cfg.train_img_size,\
        new_bboxes[...,2]*cfg.train_img_size,new_bboxes[...,3]*cfg.train_img_size],axis=-1)
    new_cls_id = tf.gather_nd(cls_id,postive_indices)
    return image,(new_bboxes,new_cls_id)


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, n_boxes, [x1, y1, x2, y2, class, best_anchor])
    # y_true_out: (N, grid_x, grid_y, anchors, [x1, y1, x2, y2, obj, class])

    N = tf.shape(y_true)[0]
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


@tf.function
def transform_targets(y_train):
    # y_train : (N, n_boxes, [x1, y1, x2, y2, cls])
    # y_outs: [(N, grid_x, grid_y, anchors, [x1, y1, x2, y2, obj, class])]*3, -> grid points low to high
    y_outs = []
    grid_size = cfg.train_img_size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(cfg.yolo_anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),(1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)
    for anchor_idxs in cfg.yolo_anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_labels(image, labels):
    # labels = (N,[xc,yc,w,h,cls])
    labels = tf.concat([labels[..., :2] - labels[..., 2:4] / 2.0, \
                        labels[..., :2] + labels[..., 2:4] / 2.0, \
                        labels[..., 4:]],
                        axis=-1,)
    # labels : (N,[x1,y1,x2,y2, cls_id])
    transformed_labels = transform_targets(labels)
    return image,transformed_labels
