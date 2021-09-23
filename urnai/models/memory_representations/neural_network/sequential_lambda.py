import numpy as np
import tensorflow.keras as keras
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

class SequentialLambda(keras.models.Sequential):
    def __init__(self, gamma, lamb):
        super().__init__()
        self.e_trace = []
        self.gamma = gamma
        self.lamb = lamb
        

    def train_step(self, data):
        """The logic for one training step.
        
        This method was overridden so we could interject a "modify_gradients" function before optimization.

        This allows us, for example, to modify our gradient vector with an eligibility traces vector, 
        enabling an easy and intuitive way of implementing temporal difference methods with keras.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            gradients = self.modify_gradients(gradients)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def modify_gradients(self, gradients):
        for i in range(len(gradients)):
            print("GRADIENT",i,": ")
            print(gradients[i][0])

        if len(self.e_trace) == 0: # we have not initialized e_greedy before
            for g in gradients:
                e = np.zeros(g[0].get_shape())
                self.e_trace.append(e)
        
        print(self.e_trace.shape())

        return gradients
        # if len(self.e_trace) == 0: # we have not initialized e_greedy before
        #     for g in gradients:
        #         e = np.zeros(g[0].get_shape())
        #         self.e_trace.append(e)
        # else:
        #     for i in range(len(self.e_trace)):
        #         self.e_trace[i] = self.gamma * self.lamb * self.e_trace + gradients[i] #e_trace calculation
        #         assert(self.e_trace[i].shape == gradients[i].shape)
        # return self.e_trace
