import keras
import keras.backend as K

from typing import Tuple


class Conv2D(keras.layers.Layer):
    def __init__(self,
                 filters: int = 16,
                 kernel_size: Tuple[int, int] = (3, 3),
                 pad: bool = False,
                 batch_size: int = 1,
                 **kwargs
                 ):
        self.pad = pad
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.filters = filters
        # Whether the channel axis is at the first index or at the last index in argument 'input_shape' passed
        # to functions 'build' and 'compute_output_shape'
        if K.image_data_format() == 'channels_first':
            self.channel_axis = 0
        else:
            self.channel_axis = -1
        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        print(f"input_shape: {input_shape}")
        # The input shape is a tensor '(kernel_x_dim, kernel_y_dim, channels, filters)'
        kernel_shape = self.kernel_size + (input_shape[self.channel_axis], self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer='uniform',
                                      trainable=True
                                      )

        super(Conv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        print(f"x: {x}")
        # 'conv2d' performs 2d convolution on an input x using the kernel and outputs a tensor with dimensions
        # according to the ones computed by 'compute_output_shape'. The convention 'K.image_data_format()' is 'channels_last'.
        # the 'padding' argument should be 'same' if padding is used and should be 'valid' if no padding is used
        PAD = "same" if self.pad else "valid"
        return K.conv2d(x, self.kernel, padding=PAD, data_format=K.image_data_format())
        pass

    def compute_output_shape(self, input_shape):
        shape3d = input_shape[1:]
        print(f"{self.compute_output_shape.__name__}->input_shape: {input_shape}")
        print(f"{self.compute_output_shape.__name__}->shape3d: {shape3d}")
        # If we should pad the output x_dim and output y_dim do not change from the input dims
        if self.pad:
            if K.image_data_format() == 'channels_first':
                return self.batch_size, self.filters, shape3d[1], shape3d[2]
            else:
                return self.batch_size, shape3d[0], shape3d[1], self.filters
        # otherwise we compute accordingly:
        else:
            if K.image_data_format() == 'channels_first':
                x = shape3d[1] - self.kernel_size[0] + 1
                y = shape3d[2] - self.kernel_size[1] + 1
                return self.batch_size, self.filters, x, y
            else:
                x = shape3d[0] - self.kernel_size[0] + 1
                y = shape3d[1] - self.kernel_size[1] + 1
                return self.batch_size, x, y, self.filters
