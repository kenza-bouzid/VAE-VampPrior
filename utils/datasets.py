from enum import Enum
from numpy.core.numeric import full
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from scipy.io import loadmat

mnist = tf.keras.datasets.mnist


class DatasetKey(Enum):
    MNIST = 0
    OMNIGLOT = 1
    CALTECH = 2

def get_dataset(dataset_key):
    if dataset_key == DatasetKey.MNIST:
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = (x_train.astype(np.float32) / 255)
        x_test = x_test.astype(np.float32) / 255

    elif dataset_key == DatasetKey.OMNIGLOT:
        x_train = tfds.load('omniglot', split="train")
        x_train = np.array([x['image'][:, :, 0]
                            for x in tfds.as_numpy(x_train)]).astype(np.float32) / 255
        x_test = tfds.load('omniglot', split="test")
        x_test = np.array([x['image'][:, :, 0]
                           for x in tfds.as_numpy(x_test)]).astype(np.float32) / 255
    elif dataset_key == DatasetKey.CALTECH:
        full_caltech_dict = loadmat("../data/caltech101_silhouettes_28.mat")
        full_caltech = np.reshape(full_caltech_dict['X'], [-1,28,28])
        np.random.shuffle(full_caltech)
        split = int(0.9 * full_caltech.shape[0])
        x_train = full_caltech[:split].astype(np.float32)
        x_test =  full_caltech[split:].astype(np.float32)
    np.random.shuffle(x_train)
    np.random.shuffle(x_test)
    return (x_train, x_test)