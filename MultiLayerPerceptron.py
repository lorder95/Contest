from keras import backend as K
from keras.layers import Layer
from keras.models import Sequential
import numpy as np


class PerceptronLayer(Layer):

    def __init__(self, output_dim, act='step', use_bias=False, ** kwargs):
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.act = act

        if act != 'step' and act != 'sigmoid' and act != 'softmax' and act != 'tanh':
            raise Exception("Activation function not found.")

        super(PerceptronLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if len(input_shape) > 2:
            raise Exception("Input shape must be 1D")
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='random_uniform',
                                      trainable=True)

        if self.use_bias:
            self.th_w = self.add_weight(name='th',
                                        shape=(self.output_dim,),
                                        initializer='random_uniform',
                                        trainable=True)
        else:
            self.th_w = self.add_weight(name='th',
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=False)

        # Be sure to call this at the end
        super(PerceptronLayer, self).build(input_shape)

    def call(self, x):
        w_sum = K.dot(x, self.kernel) - (self.th_w if self.use_bias else 0)

        if self.act == 'step':
            ret = w_sum - self.th_w
            return K.clip(ret, -(10**-6), 10**-6)
        elif self.act == 'sigmoid':
            return K.sigmoid(w_sum)
        elif self.act == 'softmax':
            return K.softmax(w_sum)
        elif self.act == 'tanh':
            return K.tanh(w_sum)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "act": self.act,
            "use_bias": self.use_bias}
        base_config = super(PerceptronLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    model = Sequential()
    model.add(PerceptronLayer(10, input_shape=(100,)))

    y_pred = model.predict(np.random.rand(2, 100))

    print(y_pred)
