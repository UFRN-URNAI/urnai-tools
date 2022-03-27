import os

import tensorflow as tf


# THIS FILE SHOULD ONLY BE USED WITH TF 2.0

def ignore_tensorflow_gpu():
    """
    This function tells tensorflow
    to ignore any available GPU and
    train on CPU instead.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def allow_memory_growth():
    """
    Sometimes, tensorflow complains about
    not enough memory on a GPU.
    To surpass this, we must tell it to allow
    memory growth. If you need it, call this function.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
