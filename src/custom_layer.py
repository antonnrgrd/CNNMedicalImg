from keras import backend as K
from keras.layers import Layer

class Dense(Layer):	#All custom made classes must be an instance of Layer.
    
    '''
    To make a custom layer, we will need to implement 4 functions.
    '''	

    def __init__(self, output_dim, use_bias=True, **kwargs):	#Must be extended.
        self.output_dim = output_dim		#We save the output dimension.
        self.use_bias = use_bias
        super(Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        build is called when we initialize the layer.

        we initialize the weights of the layer, and also the
        bias.

        We initialize the weights of the layer to be in an
        (input * output) tensor (which is a multi dimensional array).
        """
        self.kernel = self.add_weight(name='kernel', 		
                                      shape=(input_shape[1], 	
                                      self.output_dim), 
                                      initializer='uniform',	#Make weights random.
                                      trainable=True)		#We need trainable weights.

        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                          shape=(self.output_dim,),
                                          initializer='zeros',	#Initialize bias to be all 0's. Maybe try 'uniform'.
                                          trainable=True)	#Bias must also be trainable.

        super(Dense, self).build(input_shape)	#Have to be called in the end.


    def call(self, inputs):
  
        '''
        Call is the function used when layer takes an input and
        generates an output.

        From the documentation we see that a dense layer returns:
        'output = activation(dot(input, kernel) + bias)'
      
        (The kernel is the layers weights).

        We don't have to do the activation function in here,
        since we can just add the activation function after the layer,
        which will have the same effect. It looks like this in the main file:

            model.add(custom_layer.Dense(64))
            model.add(Activation('relu'))

        We need to utilize backends math operations (add and dot)

        '''

        result = K.dot(inputs, self.kernel)	#First we take the dot product

        if (self.use_bias):			#We then add the bias to the result.
            result = K.bias_add(result, self.bias)		#Adds bias to a tensor according to documentation. Has optional argument data_format.

        return result


    def compute_output_shape(self, input_shape):
        #The output shape will be (batch_size * output_dimension).
        return (input_shape[0], self.output_dim)





