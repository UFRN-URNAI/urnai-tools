# File tf_utils.py

## Function ignore_tensorflow_gpu()

This function tells tensorflow to ignore any available GPU and train on CPU instead.

- No Arguments.

## Function allow_memory_growth()

Sometimes, tensorflow complains about not enough memory on a GPU. To surpass this, we must tell it
to allow memory growth. If you need it, call this function.

- No Arguments.
