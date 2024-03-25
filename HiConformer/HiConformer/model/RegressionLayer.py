import numpy as np
import tensorflow as tf
from HiConformer.model.Activation_Selector import get_activation_function

class Regression_Layer(tf.keras.layers.Layer):
    def __init__(self,sizes,last_activation='relu',name=None,Dropout=0.5,Output_dim=1):
        activation            = get_activation_function()
        super(Regression_Layer, self).__init__(name=name)
        self.regression_layer = [tf.keras.layers.Dense(size) for size in sizes]
        self.Dropout_Layer    = [tf.keras.layers.Dropout(Dropout) for _ in range(len(sizes))]
        self.Output_layer     = tf.keras.layers.Dense(Output_dim, activation=last_activation)
        self.Activation_Layer = [tf.keras.layers.Activation(activation) for _ in range(len(sizes))]
        self.layernorm_layer  = [tf.keras.layers.LayerNormalization() for _ in range(len(sizes))]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'regression_layer'  : self.regression_layer,
            'Dropout_Layer'     : self.Dropout_Layer,
            'Output_layer'      : self.Output_layer,
            'Activation_Layer'  : self.Activation_Layer,
            'layernorm_layer '  : self.layernorm_layer
        })
        return config

    def call(self, input):
        for layer_num in range(len(self.regression_layer)):
            input = self.regression_layer[layer_num](input)
            input = self.layernorm_layer[layer_num](input)
            input = self.Activation_Layer[layer_num](input)
            input = self.Dropout_Layer[layer_num](input)
        input = self.Output_layer(input)
        return input
