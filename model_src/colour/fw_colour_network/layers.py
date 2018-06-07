from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2

#==========================================================================================
####### V1 
#========================================================================================== 
def v1_layer_colours (input_layer): 
    red,green,blue = tf.split(input_layer, num_or_size_splits=3, axis=3)
    return {"red":v1_layer(red), "green":v1_layer(green),"blue":v1_layer(blue)}




def v1_layer(input_layer,weights = None, bias = None): 
  v1_filter = {}
  v1_filter_params = {}
  v1  = {}
  for degrees in [0,45,90,135]:
 
    v1_filter_params[degrees] = {
    'ksize':(9, 9), 
    'sigma':0.5, #e sigma/standard deviation of the Gaussian envelope 1.0
    'theta': degrees, #represents the orientation of the normal to the parallel stripes of a 
    'lambd':15.0, #represents the wavelength of the sinusoidal factor 1.25
    'gamma':0.012 } #e spatial aspect ratio  0.015
    v1_filter[degrees] = cv2.getGaborKernel(**v1_filter_params[degrees])
    v1_filter[degrees] = tf.expand_dims(v1_filter[degrees], 2) 
    v1_filter[degrees] = tf.expand_dims(v1_filter[degrees], 3)
   
    #temp = v1_filter[degrees] 
    #v1_filter[degrees] = tf.concat([temp,temp,temp] , 3) 
    v1_filter[degrees] = tf.cast(v1_filter[degrees] , tf.float32)

    # make the filter to have 4 dimensions.

  # Apply the filter on `image`
    with tf.name_scope("v1_degree_"+str(degrees)):

      v1 [degrees] = tf.nn.conv2d(input_layer, filter = v1_filter[degrees], strides = [1,6,6,1], padding = 'SAME', name = "Gabor_Conv")
      v1 [degrees] = tf.tanh(v1[degrees],name = "v1_tanh")
      v1 [degrees] = tf.reshape(v1 [degrees],[-1,40 * 40 ,1],name = "flatten")
      

      #v1 [degrees] = tf.cast(v1[degrees],tf.float64)


 
  return v1



#================================================================================================================================================================================
#================================================================================================================================================================================
#================================================================================================================================================================================
#================================================================================================================================================================================
#HERE THE FUN STUFF WITH THE ADDED SPARSE tensors start, 
#the weights are saved as 1-d arrays as well as num_class-D array 
#more intuitive (but less efficient) is saving them as Sparse Tensors and just adding them together in this example 
#for now the more efficient way is implemented, since the problem lies in broadcasting across the batchsize
#================================================================================================================================================================================
#================================================================================================================================================================================

#previous_layer [batch_size . input_shape]

#cue: [batch_size, num_classes] 

def calculate_new_layer (previous_layer, layer_key, packed_weights, cues , input_shape = 40 * 40,inhibit_add = True):
  

  global_weights, cue_weights, indices = packed_weights
  
  
  ind_values = tf.matmul(cues,cue_weights[layer_key]) 

  if inhibit_add:
    ind_values = tf.add(ind_values,global_weights[layer_key])
    #ind_values = tf.Print(ind_values,[tf.is_nan(ind_values)],summarize = 100000)
  else: 
    ind_values = tf.tanh(ind_values)
    ind_values = tf.multiply(ind_values,global_weights[layer_key])


  def make_weight_matrix(row):
    


    layer = row[1]
    weights = row[0]
    parse_weights = tf.SparseTensor(indices = indices, values = weights, dense_shape = [input_shape,input_shape])
    out = tf.sparse_tensor_dense_matmul(sp_a = parse_weights,b = layer)
    return out

  layer_out = tf.map_fn(make_weight_matrix, (ind_values,previous_layer),dtype = tf.float32)

  return layer_out 


#===========================================================================================================  
####### V2 with CUE 
#===========================================================================================================    




def v2_layer_colours (v1, cues, packed_weights,input_shape = 40 * 40):
    v2 = {}
    for cl in ["red","green","blue"]:


      v2[cl] = v2_layer_cue(v1[cl], cues,  packed_weights[cl], input_shape = input_shape)

    return v2  


def v2_layer_cue(v1, cues,  packed_weights,input_shape = 40 * 40): 

  with tf.name_scope("v2_layer"):

    def make_v2(i_1, i_2, w_1, w_2 ,i_bias): 
      
      first_v1 = calculate_new_layer (v1[i_1], w_1, packed_weights, cues, input_shape = input_shape) 
      second_v1 = calculate_new_layer (v1[i_2], w_2, packed_weights, cues, input_shape = input_shape) 
      subtracted_v1 = tf.subtract(first_v1,second_v1, 
        name = "subtract_"+ str(i_1)+ "_"+ str(i_2))
       

      return tf.tanh(subtracted_v1)






    v2_0_90      =    make_v2(0, 90, 'w_0', 'w_90' ,'v2_0_90')
    
    v2_45_135        =      make_v2(45, 135, 'w_45', 'w_135' ,'v2_45_135')  
    return (v2_0_90, v2_45_135)


#==========================================================================================
####### V4 CUE
#==========================================================================================

def v4_layer_colours(v2,cues,  packed_weights,  input_shape = 40 * 40):
    v4 = {}
    for cl in ["red","green","blue"]:
      v2_1, v2_2 = v2[cl]
      v4[cl] = v4_layer_cue(v2_1, v2_2,cues,  packed_weights[cl], input_shape = input_shape)
    return v4


def v4_layer_cue(v2_1, v2_2,cues,  packed_weights,  input_shape = 40 * 40):


  #with tf.name_scope("v4_layer"):
  first_v2_weighted   = calculate_new_layer (v2_1, 'w_4_0_90',    packed_weights, cues, input_shape = input_shape) 
  second_v2_weighted  = calculate_new_layer (v2_2, 'w_4_45_135', packed_weights, cues, input_shape = input_shape) 

  v4 = tf.add(first_v2_weighted,second_v2_weighted) #, name = "add_2_v2")  
  v4_out= tf.tanh(v4, name = "activation_v4")

  return v4_out 



#==========================================================================================
####### PIT & AIT CUE 
#==========================================================================================
def PIT_layer_colours(v4,cues,  packed_weights,  input_shape = 40 * 40):
    PIT = {}
    for cl in ["red","green","blue"]:
      
      PIT[cl] = PIT_layer_cue (v4[cl],cues,  packed_weights[cl], input_shape = input_shape)
    return PIT


def PIT_layer_cue (v4,cues,  packed_weights, input_shape = 40 * 40):
  pit = calculate_new_layer (v4, 'w_PIT',    packed_weights, cues, input_shape =  input_shape) 
  
  return tf.tanh(pit)  



def AIT_layer_colours(PIT,number_of_classes = 4): 
  
  PIT_list = PIT.values()
  PIT_combined = tf.concat([tf.reshape(p,[-1,40 * 40]) for p in PIT_list],axis = 1)
  AIT =  tf.layers.dense(inputs=PIT_combined, units= number_of_classes + 1)

  return AIT 





'''
def AIT_layer_colours(PIT,number_of_classes = 4): 
  
  PIT_concat = tf.concat(PIT_list,2) 
  PIT_flattened = tf.reshape(PIT_concat,[-1,40 * 40 * 3])

  AIT =  tf.layers.dense(inputs=PIT_flattened, units= number_of_classes + 1)

  return AIT 
'''  

