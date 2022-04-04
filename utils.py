import os
import numpy as np
import cv2
from shutil import copytree
from config import Configuration
cfg = Configuration()


def clone_checkpoint(ckpt_dir):
    '''
    Create a copy of the existing checkpoint to avoid overwriting
    the current checkpoint. This will be handy if you cannot afford continuous
    training (as in Colab).
    '''
    new_ckpt = os.path.join('train_ckpts', os.path.split(cfg.log_dir)[-1])
    if ckpt_dir is not None:
        assert os.path.exists(ckpt_dir)
        copytree(ckpt_dir, new_ckpt)
    ckpt_dir = new_ckpt
    return ckpt_dir

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img