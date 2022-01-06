import tensorflow as tf
from config import cfg
from data.augment import random_crop

def decode_labels(labels):
    labels = tf.strings.strip(labels)
    labels = tf.strings.split(labels,'\n')
    labels = tf.strings.split(labels,' ')
    labels = tf.strings.to_number(labels)
    labels = labels.to_tensor()
    bbox = labels[:,1:]
    cls_id  = tf.cast(labels[:,0], dtype=tf.int32)
    return bbox,cls_id

def decode_images(image):
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.cast(image,tf.float32)
    return image

def pad_to_fixed_size(image,labels):
    image_shape =  tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    ratio = cfg.img_size/tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32),antialias=True)
    image = tf.image.pad_to_bounding_box(image, 0, 0, tf.cast(cfg.img_size[0],tf.int32),tf.cast(cfg.img_size[1],tf.int32))
    bbox,cls_id = labels[0],labels[1]
    bbox = tf.stack([(bbox[...,0]*image_shape[1])/cfg.img_size[1],(bbox[...,1]*image_shape[0])/cfg.img_size[0],\
        (bbox[...,2]*image_shape[1])/cfg.img_size[1],(bbox[...,3]*image_shape[0])/cfg.img_size[0]],axis=-1)
    # bbox = bbox/cfg.img_size
    return image,(bbox,cls_id)


def crop_or_pad_to_fixed_size(image,labels):
    image_shape =  tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    image,(bbox,cls_id) = tf.cond(tf.logical_or(image_shape[0]<=cfg.img_size[0],image_shape[1]<=cfg.img_size[1]),\
        lambda:pad_to_fixed_size(image,labels),lambda:random_crop(image,labels))
    return image,(bbox,cls_id)

def bbox_rescale(image,labels):
    bbox,cls_id = labels
    postive_mask = tf.logical_or(tf.logical_or(bbox[:,0]!=0.0,bbox[:,1]!=0.0),tf.logical_or(bbox[:,2]!=0.0,bbox[:,3]!=0.0))
    postive_indices = tf.where(postive_mask)
    new_bboxes = tf.gather_nd(bbox,postive_indices)
    # new_bboxes = new_bboxes*cfg.img_size
    new_bboxes = tf.stack([new_bboxes[...,0]*cfg.img_size[1],new_bboxes[...,1]*cfg.img_size[0],\
        new_bboxes[...,2]*cfg.img_size[1],new_bboxes[...,3]*cfg.img_size[0]],axis=-1)
    new_cls_id = tf.gather_nd(cls_id,postive_indices)
    return image,(new_bboxes,new_cls_id)

# Not implemented
def transform_labels(image, labels):
    transformed_labels = transform_targets(labels)
    return image,transformed_labels
