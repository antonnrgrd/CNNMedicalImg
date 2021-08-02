from keras import backend as K
import keras
from keras.utils import conv_utils
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec

class CustomPooling2D(Layer):

    def __init__(self, pool_size=(2, 2), strideSize=None, padding='valid', data_format=None, pool_mode = 'max', **kwargs):
        super(CustomPooling2D, self).__init__(**kwargs)
        # initialize variables

        #set default stride to whole pool_size (thus no overlap)
        if strideSize is None:
            strideSize = pool_size

        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strideSize = conv_utils.normalize_tuple(strideSize, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        # Argument 'pool_mode' should be either 'max' or 'avg'
        self.pool_mode = pool_mode

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding, self.strideSize[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding, self.strideSize[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def _pooling_function(self, inputs, pool_size, strides,padding, data_format):
        #calling the native pooling function of keras
        output = K.pool2d(inputs, pool_size, strides, padding, data_format, pool_mode=self.pool_mode)
        return output

    def call(self, inputs):
        output = self._pooling_function( inputs=inputs, pool_size=self.pool_size, strides=self.strideSize, padding=self.padding, data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strideStride,
                  'data_format': self.data_format}

        base_config = super(CustomPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

