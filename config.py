import os
from datetime import datetime


class Configuration():
    def __init__(self):

        # dataset config
        self.train_gt_path = 'ds/train/gt/*.png'
        self.train_input_path = 'ds/train/input/*.png'
        self.train_batch_size = 8
        self.train_img_shape = [256, 256, 3]
        self.train_augmentation = True
        self.val_gt_path = 'ds/val/gt/*.png'
        self.val_input_path = 'ds/val/input/*.png'
        self.val_batch_size = 1
        self.val_img_shape = [1024, 2048, 3]
        self.val_augmentation = False
        self.img_size = 416
        self.num_classes = 80
        self.num_anchors = 9

        # Augmentations
        self.augmentations = {
            "mosaic":False,
            "mixup":False,
            "flip_vertical":True,
            "flip_horizontal":True,
            "rotation_degree":0.0,
            "random_hue":0.5,
            "random_saturation":0.5,
            "random_value":0.5
        }

        # Anchors
        self.yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
        self.yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

        # training config
        self.ckpt_dir = None  # assign None if starting from scratch
        self.train_mode = ['best', 'last'][1]  # continue training from last epoch or epoch with best accuracy
        self.n_epochs = 10000
        self.lr_boundaries = [8500, 19000]
        self.lr_values = [1e-4, 9e-5, 1e-5]
        self.weight_mse_loss = 1

        # visualization config
        self.val_freq = 4             # frequency of validation - assign high value to accelerate training
        self.display_frequency = 50   # frequency of printing sample predictions - must be a multiple of val_freq
        self.display_samples = 5      # number of samples printed
        self.log_dir = os.path.join('logs', str(datetime.now().strftime("%d%m%Y-%H%M%S")))  # Tensorboard logging

        # Model config
        self.width = [0.50, 0.75, 1.0, 1.25]
        self.depth = [0.33, 0.67, 1.0, 1.33]
        self.version = 's'
