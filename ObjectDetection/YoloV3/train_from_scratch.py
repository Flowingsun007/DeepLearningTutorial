import tensorflow as tf

from utils.visualize import visualize_training_results
from yolo.yolo_v3 import YOLOV3
from configuration import CATEGORY_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, \
    save_model_dir, save_frequency, load_weights_before_training, load_weights_from_epoch, \
    test_images_during_training, test_images
from yolo.loss import YoloLoss
from data_process.make_dataset import generate_dataset, parse_dataset_batch
from yolo.make_label import GenerateLabel
from utils.preprocess import process_image_filenames


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def generate_label_batch(true_boxes):
    true_label = GenerateLabel(true_boxes=true_boxes, input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH]).generate_label()
    return true_label


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices(device_type="GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # dataset
    train_dataset, train_count = generate_dataset()

    net = YOLOV3(out_channels=3 * (CATEGORY_NUM + 5))
    print_model_summary(network=net)

    if load_weights_before_training:
        net.load_weights(filepath=save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    # loss and optimizer
    yolo_loss = YoloLoss()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=3000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)


    # metrics
    loss_metric = tf.metrics.Mean()

    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            yolo_output = net(image_batch, training=True)
            loss = yolo_loss(y_true=label_batch, y_pred=yolo_output)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, net.trainable_variables))
        loss_metric.update_state(values=loss)


    for epoch in range(load_weights_from_epoch + 1, EPOCHS):
        step = 0
        for dataset_batch in train_dataset:
            step += 1
            images, boxes = parse_dataset_batch(dataset=dataset_batch)
            labels = generate_label_batch(true_boxes=boxes)
            train_step(image_batch=process_image_filenames(images), label_batch=labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch,
                                                                   EPOCHS,
                                                                   step,
                                                                   tf.math.ceil(train_count / BATCH_SIZE),
                                                                   loss_metric.result()))

        loss_metric.reset_states()

        if epoch % save_frequency == 0:
            net.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')

        if test_images_during_training:
            visualize_training_results(pictures=test_images, model=net, epoch=epoch)

    net.save_weights(filepath=save_model_dir+"saved_model", save_format='tf')