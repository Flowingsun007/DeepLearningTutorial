from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import shutil
import random
import numpy as np
import cv2
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core import yolov3,dataset
from core.config import cfg

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        output = model(image_data, training=True)
        giou_loss, conf_loss, prob_loss = yolo_loss(target, output)
        total_loss = giou_loss+conf_loss+prob_loss
        # 对权重矩阵更新梯度(应用梯度下降)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))
        # 更新学习率
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # 写log
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


def yolo_loss(target, output):
    """计算损失，for循环计算三个采样率下的损失，注意：此处取三种采样率下的总损失而不是平均损失"""
    giou_loss=conf_loss=prob_loss=0
    for i in range(3):
        # pred.shape (8, 52, 52, 3, 25) -----  output[i].shape  (8, 52, 52, 75)
        pred = yolov3.decode(output[i], i)
        loss_items = yolov3.compute_loss(pred, output[i], *target[i], i)
        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
    return [giou_loss, conf_loss, prob_loss]



def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


if __name__=='__main__':
    """注意：加载darknet训练好的模型如：yolov3.weights，用utils.load_weights(model, model_path)否则直接model.load_weights即可"""

    # 构建数据集
    trainset = dataset.Dataset('train')
    # 构建log
    logdir = "./data/log"
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)
    # 构建yolov3网络
    model = yolov3.build_yolov3()
    # model.load_weights('./weight/60_epoch_yolov3_weights')
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam()
    # 训练参数
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch
    # 模型训练
    for epoch in range(cfg.TRAIN.EPOCHS):
        for image_data, target in trainset:
            train_step(image_data, target)
        model.save_weights(str(epoch+1) + "_epoch_yolov3_weights")
        # save_weights(filepath, overwrite=True, save_format=None) 参数save_format可选'h5' or 'tf', filepath后缀为.keras或.h5时存成.h5否则默认存tf格式


