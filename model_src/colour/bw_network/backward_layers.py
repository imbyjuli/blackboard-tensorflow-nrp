from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2




#helper function calculation the new backward layers with input from the upper backward layer as well as input from the forward layer

def make_backward_layer_sparse(prev_layer, forward_layers, sp_weights, indexes,bias ,  prev_shape = 40* 40, output_shape =  40 * 40): 

    bw_weight = tf.sparse_to_dense(sparse_indices = indexes,output_shape = [prev_shape, output_shape],sparse_values = sp_weights,validate_indices=False)
    #[batchsize,prev_shape,output_shape] 

    
    bw_layer =  tf.add(tf.matmul(prev_layer,bw_weight), bias) 
    output_values = tf.multiply(forward_layers, bw_layer)

    return tf.tanh(output_values)  


'''
    output_values = tf.map_fn(rf_matmul,(prev_layer,final_weights))

#helper function to calculate: the mapping of  rf matmul [batchsize, n] matmul [batchsize,n,n]
def rf_matmul(tuple): 
  #from [n,] -> [n,1] 
  x = tf.expand_dims(tuple[0],1)
  #size[n,n]
  w = tuple[1]
  # [n,n] * [n,1] -> [n,1] 
  y = tf.matmul(w,x)  
  #[n,1] -> [n,] 
  return tf.reduce_sum(y,1)
'''


#==========================================================================================
####### PIT & AIT CUE 
#==========================================================================================
def ait_backwards(input_layers, output_shape = 40*40): 
    return  tf.layers.dense(inputs= input_layers["AIT"], units = output_shape)


def pit_backwards(ait_backward,input_layers,packed_backward_weights, weights,bias, output_shape = 40*40, use_sparse =True):   
  pit_flat = input_layers["PIT"]
  if use_sparse: 
    sp_weights, indexes = packed_backward_weights 
    return make_backward_layer_sparse(ait_backward,pit_flat, sp_weights["PIT"],indexes["PIT"],bias["PIT"])  
  


  #else:   
  pit_weighted = tf.add(tf.multiply(pit_flat, weights["PIT"]), bias["PIT"]) 
  pit_added = tf.add(pit_weighted , ait_backward)
  pit_dense = tf.layers.dense(inputs = pit_added,units = output_shape)
  return pit_dense


#==========================================================================================
####### V4 CUE
#==========================================================================================






def v4_backwards(pit_backward,input_layers,packed_backward_weights,weights,bias, output_shape = 40*40,  use_sparse =False): 
  
  v4_flat = input_layers["v4"]#dynamical size 

  if use_sparse: 
    sp_weights, indexes = packed_backward_weights 
    return make_backward_layer_sparse(pit_backward, v4_flat, sp_weights["v4"],indexes["v4"],bias["v4"])  

  v4_weighted = tf.add(tf.multiply(v4_flat, weights["v4"]), bias["v4"]) 
  v4_added = tf.add(v4_weighted , pit_backward)
  v4_dense = tf.layers.dense(inputs = v4_added ,units = output_shape)
  return v4_dense





#===========================================================================================================  
####### V2 with CUE 
#===========================================================================================================    


def v2_backwards(v4_backward,input_layers,packed_backward_weights,weights,bias, output_shape = 40*40,  use_sparse =False): 
  
  v2_1_flat = input_layers["v2_1"]
  v2_2_flat = input_layers["v2_2"]

  if use_sparse: 
    sp_weights, indexes = packed_backward_weights 
    v2_1 =  make_backward_layer_sparse(v4_backward,v2_1_flat, sp_weights["v2_1"],indexes["v2_1"],bias["v2_1"])
    v2_2 =  make_backward_layer_sparse(v4_backward,v2_2_flat, sp_weights["v2_2"],indexes["v2_2"],bias["v2_2"])
    #TODO: better solution for the split between the two v2 layers 
    v2_full = tf.add(v2_1,v2_2)
    return tf.divide(v2_full,2.)





  v2_1_weighted = tf.add(tf.multiply(v2_1_flat, weights["v2_1"]), bias["v2_1"])
  v2_2_flat = input_layers["v2_2"]
  v2_2_weighted = tf.add(tf.multiply(v2_2_flat, weights["v2_2"]), bias["v2_2"])
  v2_backward = tf.add(v2_1_weighted , pit_dense)
  v2_backward = tf.add(v2_2_weighted , v2_backward)
  return v2_backward







#================================================================================================================================================================================
#================================================================================================================================================================================
#================================================================================================================================================================================
####### Custom loss function
#================================================================================================================================================================================
#================================================================================================================================================================================
def eucledian_distance(x,x_,y,y_): 
    x_difference = tf.subtract(x,x_)
    x_squared = tf.square(x_difference)

    y_difference = tf.subtract(y,y_)
    y_squared = tf.square(y_difference)

    sum_ = x_squared +  y_squared 
    #size vector [200,1] 
    final_vector = tf.sqrt(sum_) 
    batch_size = tf.shape(final_vector)[0] 
    loss =  tf.reduce_sum(final_vector) 
    norm_loss = tf.divide(loss, batchsize)
    return norm_loss 


#reading in the coordinates of flat image and returning one of the 4 directions (top,bottom) X (left,right)
def create_direction_for_eval(labels, row_length = 40): 
  
  max_val = tf.argmax(labels,1)
  cut_off_1 = int(row_length/2)
  cut_off_2 = row_length * cut_off_1
  

  #checks if top or bottom 
  added_ = tf.greater_equal(max_val,cut_off_2)
  y_dir = tf.cast(added_,tf.int32)


  #checks if on left or right part of image
  split = tf.greater_equal(tf.floormod(max_val,row_length), cut_off_1)
  x_dir = tf.cast(split,tf.int32)
  return x_dir +  y_dir 



'''  ait_backward = tf.layers.dense(inputs= input_layers["AIT"], units = (40 *  40))
  
  pit_flat = input_layers["PIT"] #dynamical size 

  pit_weighted = tf.add(tf.multiply(pit_flat, weights["PIT"]), bias["PIT"])
  pit_added = tf.add(pit_weighted , ait_backward)
  pit_dense = tf.layers.dense(inputs = pit_added,units = (40*40))





  v4_flat = input_layers["v4"]#dynamical size 
  v4_weighted = tf.add(tf.multiply(v4_flat, weights["v4"]), bias["v4"])
  
  v4_added = tf.add(v4_weighted , pit_dense)
  v4_dense = tf.layers.dense(inputs = v4_added ,units = (40*40))




  v2_1_flat = input_layers["v2_1"]
  v2_1_weighted = tf.add(tf.multiply(v2_1_flat, weights["v2_1"]), bias["v2_1"])

  v2_2_flat = input_layers["v2_2"]
  v2_2_weighted = tf.add(tf.multiply(v2_2_flat, weights["v2_2"]), bias["v2_2"])

  v2_backward = tf.add(v2_1_weighted , v4_dense)
  v2_backward = tf.add(v2_2_weighted , v2_backward)





  #  final_dense = tf.layers.dense(v2_backward)
  logits = tf.layers.dense(inputs = v2_backward, units = 4)
'''