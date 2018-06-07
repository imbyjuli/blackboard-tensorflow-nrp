import numpy as np
import tensorflow as tf
import itertools 


#easiest way to get the number of values to calculate: 
#extend to 3 dimensions

def get_indices (input_shape = 40 * 40, row_length = 40, k_size = 3): 
  rd = range(k_size)
  def make_row(row_i): 
    it_mask_pos = (list(itertools.product(rd,rd)))
    return  [[row_i, row_i + r + row_length * d]  for (r,d) in  it_mask_pos if row_i+r+row_length*d<input_shape and (row_i % row_length) + r < row_length ] 
  indicies = [make_row(i)  for i in range(input_shape)]

  #flatten index list 
  indicies_2 = [item for sublist in indicies for item in sublist]
  return indicies_2


##=====================================================================================================================
##=  Sparse weights
##=====================================================================================================================
def init_all_weights (input_shape, rw_length = 40 ,num_classes = 4, receptive_field_size = 3):
  packed_weights = {}

  for cl in ["red","green","blue"]: 
       global_weights, cue_weights , indices= init_all_weights_1D (input_shape, receptive_field_size = receptive_field_size)
       packed_weights[cl] = global_weights , cue_weights , indices
  return  packed_weights



def init_all_weights_1D (input_shape, rw_length = 40 ,num_classes = 4, receptive_field_size = 3):

  indices = get_indices (input_shape = input_shape, row_length =rw_length, k_size = receptive_field_size)
  values_length = len(indices)
  custom_s = 1
 
  custom_dtype = tf.float32
  with tf.name_scope("global_weights"): 
    global_weights = {
        'w_0'   : tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype = custom_dtype)) 
        ,
      'w_90'  :  tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype = custom_dtype)) 
        ,
      'w_45'  :  tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype = custom_dtype)) 
        ,
      'w_135' :   tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype = custom_dtype))  
        ,

      'w_4_0_90'  :  tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype = custom_dtype)) 
        , 
      'w_4_45_135':  tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype = custom_dtype))
       , 
      'w_PIT'     :  tf.Variable(tf.random_normal(shape = [values_length], stddev = custom_s ,dtype =  custom_dtype)) 

     
    }


  with tf.name_scope("cue_weights"): 
    cue_weights = {
        'w_0'   : tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype)) 
        ,
      'w_90'  :  tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype)) 
        ,
      'w_45'  :  tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype)) 
        ,
      'w_135' :   tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype))  
        ,

      'w_4_0_90'  :  tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype)) 
        , 
      'w_4_45_135':  tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype))
       , 
      'w_PIT'     :  tf.Variable(tf.random_normal(shape = [num_classes,values_length], stddev = custom_s ,dtype = custom_dtype)) 

     
    }








    return global_weights, cue_weights, indices









def initiate_backward_weights (input_layers_size):
  weights ={}
  bias = {}
  for  name, size in input_layers_size.items(): 
    x, y = size
    weights[name] = tf.Variable(tf.random_normal(shape = [x * y])) 
    bias [name] = tf.Variable(tf.random_normal(shape = [x * y] ))
  return weights, bias 

