from enum import Enum
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

mnist = tf.keras.datasets.mnist


class DatasetKey(Enum):
    MNIST = 0
    OMNIGLOT = 1


def get_dataset(dataset_key):
    if dataset_key == DatasetKey.MNIST:
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = (x_train.astype(np.float32) / 255)[:1000]
        x_test = x_test.astype(np.float32) / 255

    elif dataset_key == DatasetKey.OMNIGLOT:
        x_train = tfds.load('omniglot', split="train")
        x_train = np.array([x['image'][:, :, 0]
                            for x in tfds.as_numpy(x_train)]).astype(np.float32) / 255
        x_test = tfds.load('omniglot', split="test")
        x_test = np.array([x['image'][:, :, 0]
                           for x in tfds.as_numpy(x_test)]).astype(np.float32) / 255
    return (x_train, x_test)