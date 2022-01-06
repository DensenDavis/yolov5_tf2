import tensorflow as tf
from data.data_utils import convert_to_corners,convert_to_xywh,compute_iou,zero_pad_bboxes,zero_pad_bboxes_batch
import numpy as np
from config import Configuration
cfg = Configuration()

#Update the code without if condition
def random_crop(image,labels,new_h=-1.0,new_w=-1.0):
    """[Generates random crops of the Image]

    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns the cropped image and corresponding labels normalised format and class ids]
    """    

    boxes,cls_id = labels
    shape = tf.cast(tf.shape(image),tf.float32)
    org_h,org_w = shape[0],shape[1]
    boxes = tf.stack([boxes[:,0]*org_w,boxes[:,1]*org_h,boxes[:,2]*org_w,boxes[:,3]*org_h],axis=-1)
    boxes = convert_to_corners(boxes)
    new_h,new_w = tf.cond(tf.logical_and(new_h !=-1.0,new_w !=-1.0),lambda:(new_h,new_w),lambda:(cfg.img_size[0],cfg.img_size[1]))
    #Crop coordinate
    left = tf.random.uniform([],0, tf.cast(org_w,tf.int32) - tf.cast(new_w,tf.int32),tf.int32)
    right = left + tf.cast(new_w,tf.int32)
    top = tf.random.uniform([],0, tf.cast(org_h,tf.int32) - tf.cast(new_h,tf.int32),tf.int32)
    bottom = top + tf.cast(new_h,tf.int32)
    crop = tf.stack([left, top, right, bottom],axis=-1)[tf.newaxis,...]
    crop = tf.cast(crop,tf.float32)
    # Calculate IoU  between the crop and the bounding boxes
    intersection,_ = compute_iou(crop, boxes) # Incorporate it with vectorisation
    intersection = tf.squeeze(intersection,axis=0)
    # If not a single bounding box has a IoU of greater than the minimum, try again
    #Crop
    im_shape = [left, top, right, bottom]
    new_image = image[im_shape[1]:im_shape[3], im_shape[0]:im_shape[2],:] #(new_h, new_w,3)
    intersection_true = intersection>0.2
    #take matching bounding box
    indices = tf.where(intersection_true)
    new_boxes = tf.gather_nd(boxes,indices)
    #take matching labels
    new_labels = tf.gather_nd(cls_id,indices)
    #Use the box left and top corner or the crop's
    new_boxes_1 = tf.maximum(new_boxes[..., :2],crop[:,:2])
    new_boxes_1 = tf.stack([new_boxes_1[...,0]-crop[:,0],new_boxes_1[...,1]-crop[:,1]],axis=-1)
    new_boxes_2 = tf.minimum(new_boxes[..., 2:], crop[:,2:])
    new_boxes_2 = tf.stack([new_boxes_2[...,0]-crop[:,0],new_boxes_2[...,1]-crop[:,1]],axis=-1)
    new_boxes = tf.stack([new_boxes_1[:,0],new_boxes_1[:,1],new_boxes_2[:,0],new_boxes_2[:,1]],axis=-1)
    new_boxes = new_boxes/tf.stack([new_w,new_h,new_w,new_h],axis=-1)
    new_boxes = convert_to_xywh(new_boxes)
    return new_image, (new_boxes, new_labels)


def flip_horizontal(image,labels):
    """Flips the image and lables horizontally
    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns the cropped image and corresponding labels normalised format and class ids]
    """    
    bbox,cls_id = labels
    image = tf.image.flip_left_right(image)
    bbox = convert_to_corners(bbox)
    bbox = tf.stack([1-bbox[:, 2], bbox[:, 1], 1-bbox[:, 0], bbox[:, 3]],axis=-1)
    bbox = convert_to_xywh(bbox)
    return image,(bbox,cls_id)

def flip_vertical(image,labels):
    """Flips the image and lables horizontally
    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns the cropped image and corresponding labels normalised format and class ids]
    """    
    bbox,cls_id = labels
    image = tf.image.flip_up_down(image)
    bbox = convert_to_corners(bbox)
    bbox = tf.stack([bbox[:, 0], 1-bbox[:, 3], bbox[:, 2], 1-bbox[:, 1]],axis=-1)
    bbox = convert_to_xywh(bbox)
    return image,(bbox,cls_id)

def image_rotate(image,angle):
    """A PIL(Pillow) object of the image is created PIL has the feature to rotate the
      image and resize the image to make sure that the rotated image fits inside the image.

    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        angle ([int]): [angle in degress]

    Returns:
        [tf.float32,tf.float32]: [Returns the rotated image and ratio between the new shape and the orginal shape]
    """    
    h,w = image.shape[0:2]
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = image.rotate(angle,expand=True)
    image = tf.keras.preprocessing.image.img_to_array(image)
    # image = image/255.0 #Check whether this line is required or not
    n_h,n_w = image.shape[0:2]
    scale_x = n_w / w
    scale_y = n_h / h
    image = tf.image.resize(image,[h,w],method='nearest',antialias=True) #Any resize method works fine here
    ratio = tf.stack([scale_x, scale_y, scale_x, scale_y],axis=-1)
    return image,ratio

def random_rotate(image,labels):
    """[summary]

    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns the rotated image and corresponding labels normalised format and class ids]
    """    
    boxes,cls_id = labels
    boxes = convert_to_corners(boxes)
    shape = tf.cast(tf.shape(image),tf.float32)
    boxes = tf.stack([boxes[:,0]*shape[1],boxes[:,1]*shape[0],boxes[:,2]*shape[1],boxes[:,3]*shape[0]],axis=-1)
    cx = shape[1]/2
    cy = shape[0]/2
    degree = tf.random.uniform([],maxval=360,dtype=tf.int32)
    degree = tf.cast(degree,tf.float32)
    angle = degree*np.pi/180.0
    image,ratio = tf.py_function(image_rotate,[image,degree],[tf.float32,tf.float32])
    alpha = tf.math.cos(angle)
    beta = tf.math.sin(angle)
    AffineMatrix = tf.convert_to_tensor([[alpha, beta, (1-alpha)*cx - beta*cy],[-beta, alpha, beta*cx + (1-alpha)*cy]])
    box_width = tf.reshape(boxes[:,2] - boxes[:,0],shape=[-1,1])
    box_height = tf.reshape(boxes[:,3] - boxes[:,1],shape=[-1,1])
    #Get corners for boxes
    x1 = tf.reshape(boxes[:,0],[-1,1])
    y1 = tf.reshape(boxes[:,1],[-1,1])
    x2 = x1 + box_width
    y2 = y1 
    x3 = x1
    y3 = y1 + box_height
    x4 = tf.reshape(boxes[:,2],[-1,1])
    y4 = tf.reshape(boxes[:,3],[-1,1])
    corners = tf.stack([x1,y1,x2,y2,x3,y3,x4,y4], axis= -1)
    corners = tf.reshape(corners,[-1, 8])    #Tensors of dimensions (#objects, 8)
    corners = tf.reshape(corners,[-1, 2]) #Tensors of dimension (4* #objects, 2)
    corners = tf.concat([corners, tf.ones_like(corners[:,0])[...,tf.newaxis]],axis=-1) #(Tensors of dimension (4* #objects, 3)
    cos = tf.abs(AffineMatrix[0, 0])
    sin = tf.abs(AffineMatrix[0, 1])

    nW = (shape[0] * sin) + (shape[1] * cos)
    nH = (shape[0] * cos) + (shape[1] * sin)
    AffineMatrix = AffineMatrix + tf.convert_to_tensor([[0.,0.,(nW/2)-cx],[0.,0.,(nH/2)-cy]])


    #Apply affine transform
    rotate_corners = tf.transpose(tf.linalg.matmul(AffineMatrix,corners,transpose_b=True))
    rotate_corners = tf.reshape(rotate_corners,[-1,8])
    x_corners = tf.stack([rotate_corners[:,0],rotate_corners[:,2],rotate_corners[:,4],rotate_corners[:,6]],axis=-1)
    y_corners = tf.stack([rotate_corners[:,1],rotate_corners[:,3],rotate_corners[:,5],rotate_corners[:,7]],axis=-1)

    # #Get (x_min, y_min, x_max, y_max)
    x_min = tf.reduce_min(x_corners, axis= 1)
    x_min = tf.reshape(x_min,[-1, 1])
    y_min = tf.reduce_min(y_corners, axis= 1)
    y_min = tf.reshape(y_min,[-1, 1])
    x_max = tf.reduce_max(x_corners, axis= 1)
    x_max = tf.reshape(x_max,[-1, 1])
    y_max = tf.reduce_max(y_corners, axis= 1)
    y_max = tf.reshape(y_max,[-1, 1])

    new_boxes = tf.concat((x_min, y_min, x_max, y_max), axis= -1)
    new_boxes = new_boxes/ratio
    x_min = tf.clip_by_value(new_boxes[:,0],0,shape[1])/shape[1]
    y_min = tf.clip_by_value(new_boxes[:,1],0,shape[0])/shape[0]
    x_max = tf.clip_by_value(new_boxes[:,2],0,shape[1])/shape[1]
    y_max = tf.clip_by_value(new_boxes[:,3],0,shape[0])/shape[0]
    new_boxes = tf.stack((x_min, y_min, x_max, y_max), axis= -1)
    new_boxes = convert_to_xywh(new_boxes)
    return image,(new_boxes,cls_id)


def cut_mix(images,labels):
    """[summary]

    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns an image batch with one of them a mosaic of randomly cropped images from 
        that batch. The labels and class ids are adjusted accordingly.]
    """ 
    num_crops = 4
    bboxes,cls_ids = labels
    n_imgs = tf.shape(images)[0]
    crop_h = 2*cfg.img_size[0]/num_crops  # 200 = img height
    crop_w = 2*cfg.img_size[1]/num_crops  # 200 = img width
    norm_crop_dim = 1/(num_crops//2)
    mosaic_rows, final_boxes, final_cls_ids = [], [], []
    count = 0
    for i in range(num_crops//2):
        img_row = []
        for j in range(num_crops//2):
            idx = (count)%n_imgs
            img_crop, (new_boxes, new_class_ids) = random_crop(images[idx],(bboxes[idx],cls_ids[idx]),crop_h,crop_w)
            img_row.append(img_crop)
            new_boxes = new_boxes/(num_crops//2)
            new_x = norm_crop_dim*j+new_boxes[...,0]
            new_y = norm_crop_dim*i+new_boxes[...,1]
            new_boxes = tf.stack([new_x,new_y,new_boxes[...,2],new_boxes[...,3]],axis=-1)
            final_boxes.append(new_boxes)
            final_cls_ids.append(new_class_ids)
            count += 1
        mosaic_rows.append(tf.concat(img_row, axis=1))

    # merge rows of cut-mosaics
    mosaic_img = tf.concat(mosaic_rows, axis=0)
    mosaic_boxes = tf.concat(final_boxes, axis=0)
    mosaic_cls_ids = tf.concat(final_cls_ids, axis=0)
    mosaic_img,(mosaic_boxes,mosaic_cls_ids) = zero_pad_bboxes(mosaic_img, (mosaic_boxes,mosaic_cls_ids))
    
    # concat cut_mosaic image to image batch
    images = tf.concat([images,mosaic_img[tf.newaxis,:]], axis=0)
    bboxes = tf.concat([bboxes,mosaic_boxes[tf.newaxis,:]], axis=0)
    cls_ids = tf.concat([cls_ids,mosaic_cls_ids[tf.newaxis,:]], axis=0)
    return images,(bboxes,cls_ids)


def mixup(images,labels):
    """[summary]

    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns an image batch with one of them a mixup of two images from 
        that batch. The labels and class ids are adjusted accordingly.]
    """ 
    bboxes,cls_ids = labels
    img_1 = images[0] # currently first two images are used for blending
    img_2 = images[1]
    rate = tf.random.uniform([],0.2, 0.8) # blending factor
    mixup_img = img_1*rate + img_2*(1-rate)
    mixup_boxes = tf.concat([bboxes[0], bboxes[-1]], axis=0)
    mixup_cls_ids = tf.concat([cls_ids[0], cls_ids[-1]], axis=0)
    
    images,(bboxes,cls_ids) = zero_pad_bboxes_batch(images, labels, max_count=2000)
    images = tf.concat([images,mixup_img[tf.newaxis,:]], axis=0)
    bboxes = tf.concat([bboxes,mixup_boxes[tf.newaxis,:]], axis=0)
    cls_ids = tf.concat([cls_ids,mixup_cls_ids[tf.newaxis,:]], axis=0)
    return images,(bboxes,cls_ids)

def basic_augmentation(image,labels):
    """Basic augmentation techniques

    Args:
        image ([tf.float32]): [The image is of float32 with value range between 0-1]
        labels ([tuple]): [its a tuple of bboxes and class ids bboxes are normalised to 0-1 and dtype is tf.float32
        class ids are of dtype tf.int32]

    Returns:
        [tf.float32,(tf.float32,tf.int32)]: [Returns randomly tramsformed input image(in terms od brightness staurationetc.) 
        as ouput. The labels and class ids are adjusted accordingly.]
    """ 
    image = tf.cast(image,tf.uint8)
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.stateless_random_brightness(image,0.2,tf.random.uniform(shape=[2],maxval=10,dtype=tf.int32))
    image = tf.image.stateless_random_contrast(image,0.5, 2.0,tf.random.uniform(shape=[2],maxval=10,dtype=tf.int32))
    image = tf.image.stateless_random_hue(image, 0.5, tf.random.uniform(shape=[2],maxval=10,dtype=tf.int32))
    image = tf.image.stateless_random_saturation(image,0.75, 1.25,tf.random.uniform(shape=[2],maxval=10,dtype=tf.int32))
    image = tf.image.convert_image_dtype(image,tf.uint8,saturate=True)
    image = tf.cast(image,tf.float32)
    return image,labels


def apply_augmentations(image, labels):
    if cfg.augmentations["flip_horizontal"]:
        image,labels = flip_horizontal(image,labels)
    if cfg.augmentations["flip_vertical"]:
        image,labels = flip_vertical(image,labels)  
    return image, labels