import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from model import YOLOv5
from config import Configuration
from data.dataset import Dataset
from train_loop import TrainLoop
from utils import clone_checkpoint
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as lr_decay
cfg = Configuration()

dataset = Dataset()
lr_schedule = lr_decay(
    boundaries=[i*dataset.num_train_batches for i in cfg.lr_boundaries],
    values=cfg.lr_values)

model = YOLOv5('s')
_ = model(tf.random.normal((cfg.train_batch_size,cfg.train_img_size,cfg.train_img_size,3),dtype=tf.dtypes.float32), training=True)
print(model.summary())
optimizer = tf.keras.optimizers.Adam(lr_schedule)
tb_writer = tf.summary.create_file_writer(cfg.log_dir)
train_obj = TrainLoop(dataset, model, optimizer)

ckpt = tf.train.Checkpoint(
    model = train_obj.model,
    optimizer = train_obj.optimizer,
    train_dataset=train_obj.dataset.train_ds,
    epoch = tf.Variable(0, dtype=tf.dtypes.int64),
    map = tf.Variable(0.0))

ckpt_dir = clone_checkpoint(cfg.ckpt_dir)
chkpt_best = os.path.join(ckpt_dir,'best')
chkpt_best = tf.train.CheckpointManager(ckpt, chkpt_best, max_to_keep=1, checkpoint_name='ckpt')
chkpt_last = os.path.join(ckpt_dir,'last')
chkpt_last = tf.train.CheckpointManager(ckpt, chkpt_last, max_to_keep=1, checkpoint_name='ckpt')

if cfg.train_mode == 'best':
    ckpt.restore(chkpt_best.latest_checkpoint)
elif cfg.train_mode == 'last':
    ckpt.restore(chkpt_last.latest_checkpoint)
else:
    raise Exception('Error! invalid training mode, please check the config file.')

print(f"Initiating training from epoch {ckpt.epoch.numpy()}")
print("***Run tensorboard to check training metrics***")
print(f'best_map= {ckpt.map.numpy()}')

while(ckpt.epoch < cfg.n_epochs):

    ckpt.epoch.assign_add(1)
    train_obj.train_one_epoch(ckpt.epoch)
    if (ckpt.epoch % cfg.val_freq) == 0:

        save_prediction = (ckpt.epoch % cfg.display_frequency) == 0
        display_batch = train_obj.run_validation(save_prediction, ckpt.epoch.numpy())

        # print(f'step={ckpt.epoch}, {train_obj.mean_train_loss.result()}, {train_obj.mean_train_loss.result()}')
        # with tb_writer.as_default():
        #     tf.summary.scalar('mean_train_loss', train_obj.mean_train_loss.result(), step=ckpt.epoch)  # Average of stepwise loss of that epoch
        #     tf.summary.scalar('mean_val_loss', train_obj.mean_val_loss.result(), step=ckpt.epoch)
            # if(save_prediction):
            #     tf.summary.image("val_images", display_batch, step=ckpt.epoch, max_outputs=cfg.display_samples) 

        # if ckpt.max_psnr <= train_obj.val_psnr.result():
        #     ckpt.max_psnr.assign(train_obj.val_psnr.result())
        #     chkpt_best.save(checkpoint_number=1)
    chkpt_last.save(checkpoint_number=1)
    print(f'mean_train_loss : {train_obj.mean_train_loss.result().numpy()}')
    print(f'mean_val_loss : {train_obj.mean_val_loss.result().numpy()}')
    # print(f'psnr : best/last = {ckpt.max_psnr.numpy()}/{train_obj.val_psnr.result().numpy()}')
