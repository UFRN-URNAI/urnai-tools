import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

class SequentialLambda(keras.models.Sequential):
    def __init__(self, gamma, lamb):
        super().__init__()
        self.e_trace = []
        #self.e_trace = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        self.gamma = gamma
        self.lamb = lamb

    @tf.function
    def train_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            # compute gradients
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            gradients = self.modify_gradients(gradients)
            # update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        # update metrics (loss, etc)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # return dict with current metrics
        return {m.name: m.result() for m in self.metrics}

    def modify_gradients(self, gradients):
        if len(self.e_trace) == 0: # if we have not initialized e_trace before
            self.e_trace = gradients
            # for g, i in enumerate(gradients):
            #     zero_tensor = tf.zeros(tf.shape(g))
            #     self.e_trace.write(i, zero_tensor)

        # we always want to diminish e_trace (which is basically our gradients)
        # by self.gamma*self.lamb to achieve the eligibility traces algorithm
        for i in range(len(self.e_trace)):
                self.e_trace[i] = self.gamma * self.lamb * self.e_trace[i] + gradients[i] #e_trace calculation
                assert(self.e_trace[i].shape == gradients[i].shape)
        return self.e_trace
    
    def reset_e_trace(self):
        self.e_trace = []
