import datetime
import numpy as np
import tensorflow as tf
from network import NetworkInNetwork

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class DataLoader():
    def __init__(self):
        initial_data = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = initial_data.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32)/255.0, axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32)/255.0, axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to input shape
        resized_images = tf.image.resize_with_pad(self.train_images[index],64,64,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to input shape
        resized_images = tf.image.resize_with_pad(self.test_images[index],64,64,)
        return resized_images.numpy(), self.test_labels[index]


def train_nin(batch_size, epoch):
    dataLoader = DataLoader()
    # create and compile model
    model = NetworkInNetwork.build_networkinnetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.8, nesterov=False)
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # train
    num_iter = dataLoader.train_images.shape[0] // batch_size
    for e in range(epoch):
        for i in range(num_iter):
            train_images, train_labels = dataLoader.get_batch_train(batch_size)
            model.fit(train_images, train_labels,
                epochs=1,
                shuffle=False
            )
        model.save_weights(str(e+1) + '_epoch_networkinnetwork_weight.h5')

def test_nin(weight_path, batch_size):
    dataLoader = DataLoader()
    test_images, test_labels = dataLoader.get_batch_test(batch_size)
    model = NetworkInNetwork.build_networkinnetwork()
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build((1,64,64,1))
    model.load_weights(weight_path)
    model.evaluate(test_images, test_labels, verbose=2)


if __name__ == '__main__':
    # 训练
    # train_nin(256, 20)

    # 测试
    test_nin('./weight/20_epoch_networkinnetwork_weight.h5', 10000)  # acc 90.7