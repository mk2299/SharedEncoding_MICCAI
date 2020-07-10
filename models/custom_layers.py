import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer



class CrowdsRegression_SRM_conv_blocks(Layer):

	def __init__(self, n_outputs = (111,127,111), kernel_size = (3,3,3), num_subs = 10,  num_features = 10, conn_type="VW+B", **kwargs):
		self.num_subs = num_subs
		self.conn_type = conn_type
		self.n_outputs =  n_outputs
		self.num_features =  num_features
		self.kernel_size =  kernel_size
		super(CrowdsRegression_SRM_conv_blocks, self).__init__(**kwargs)

############## Hard coded for specific channels dimensions ############
####### [ADD FLEXIBLE UPDATES LATER] #######################
        
	def build(self, input_shape):
		self.kernel = []
        
		self.kernel.append(self.add_weight("CrowdLayer", (*self.kernel_size ,128, 256, self.num_subs),
								  initializer=keras.initializers.glorot_uniform(),
								  trainable=True))
		self.kernel.append(self.add_weight("CrowdLayer", (128,  self.num_subs),
								  initializer=keras.initializers.glorot_uniform(),
								  trainable=True))

		self.kernel.append(self.add_weight("CrowdLayer", (*self.kernel_size , 1, 128, self.num_subs),
								  initializer=keras.initializers.glorot_uniform(),
								  trainable=True))
		self.kernel.append(self.add_weight("CrowdLayer", (1,  self.num_subs),
								  initializer=keras.initializers.glorot_uniform(),
								  trainable=True))        
        

		super(CrowdsRegression_SRM_conv_blocks, self).build(input_shape)  # Be sure to call this somewhere!

##### Hard-coded for final output dimensions to match MNI 1.6mm ###########  

####### [ADD FLEXIBLE UPDATES LATER] #######################

	def call(self, x):

		out = []
		for r in range(self.num_subs): 
				temp = tf.nn.bias_add(tf.nn.conv3d_transpose(x, self.kernel[0][:,:,:,:,:,r],  output_shape = tf.constant([1, 55, 63, 55, 128]), strides = (1,2,2,2,1), padding = "VALID"), self.kernel[1][:,r])   
				temp = K.elu(temp)
				temp = tf.nn.bias_add(tf.nn.conv3d_transpose(temp, self.kernel[2][:,:,:,:,:,r],  output_shape = tf.constant([1, 111, 127, 111, 1]), strides = (1,2,2,2,1), padding = "VALID"), self.kernel[3][:,r])   
				temp = K.elu(temp)                
				out.append(temp)
		res = tf.stack(out)
		print(res.shape)        
		res = tf.transpose(res[:,:,:,:,:,0], [1, 2,3,4, 0])
            

		return res

	def compute_output_shape(self, input_shape):
		return (input_shape[0], *self.n_outputs, self.num_subs)
    