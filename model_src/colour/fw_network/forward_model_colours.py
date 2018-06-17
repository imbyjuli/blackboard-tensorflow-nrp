from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2




import csv

from print_layers import print_activation_dict,tensor_to_colour
from weights import  init_all_weights
from layers import v1_layer_colours,v2_layer_colours,v4_layer_colours,PIT_layer_colours,AIT_layer_colours

tf.logging.set_verbosity(tf.logging.INFO)





# Our application logic will be added here

def ventral_feed_forward(features, labels, mode):
#==========================================================================================
####### weights and biases described in v_layers_weights
#========================================================================================== 

  n_field_1 = 40 * 40


  
  packed_weights = init_all_weights(n_field_1)
#==========================================================================================
####### initiation
#========================================================================================== 

  image_size = [240,240]
  img_x,img_y =  image_size 
  features_float  = tf.cast(features["x"], tf.float32)
  input_layer = tf.reshape(features_float, [-1, img_x,img_y , 3]) #almost unneccessary 
  cues = features ["cues"]



#==========================================================================================
####### layers described in v_layers
#========================================================================================== 
  
  v1 = v1_layer_colours(input_layer)
  v2 = v2_layer_colours(v1,cues, packed_weights) 
  v4  = v4_layer_colours(v2, cues, packed_weights)
  #pit = v4
  PIT  = PIT_layer_colours(v4,cues, packed_weights)

  AIT = AIT_layer_colours (PIT,4)#change number to 16 if different colour is desired













#==========================================================================================
####### print data to board: 
#========================================================================================== 

  #'''
  if mode == tf.estimator.ModeKeys.TRAIN:
    output_num = 1 
    for cl in ["red","green","blue"]: 
      cl_n = "_"+cl 
      for degrees in [0,45,90,135]:
        tf.summary.image("v1_"+str(degrees)+cl_n , tensor_to_colour( tf.reshape(v1[cl][degrees],[-1,40,40,1])), output_num)

      tf.summary.image("v2_1"+cl_n,  tensor_to_colour(tf.reshape(v2[cl][0],[-1,40,40,1])), output_num)
      tf.summary.image("v2_2"+cl_n,  tensor_to_colour(tf.reshape(v2[cl][1],[-1,40,40,1])), output_num)
      tf.summary.image("v4"+cl_n,  tensor_to_colour(tf.reshape(v4[cl],[-1,40,40,1])), output_num)    
      tf.summary.image("pit"+cl_n, tensor_to_colour(tf.reshape(PIT[cl],[-1,40,40,1])   ), output_num)
    tf.summary.image("cue",tf.reshape(cues,[-1,2,2,1]),output_num)



#==========================================================================================
####### Prediction with Tensorflow 
#========================================================================================== 







  if mode == tf.estimator.ModeKeys.PREDICT:

    predictions = {
      "classes": tf.argmax(input=AIT, axis=1),

      "AIT": AIT, 
      "probabilities": tf.nn.softmax(AIT, name="softmax_tensor")
    }

    for cl in ["red","green","blue"]: 
      predictions[cl +  "_v2_1"] = tf.reshape(v2[cl][0],[-1,40,40,1])
      predictions[cl +  "_v2_2"] =  tf.reshape(v2[cl][1],[-1,40,40,1])
      predictions[cl +  "_v4"] =  tf.reshape(v4 [cl],[-1,40,40,1])
      predictions[cl +  "_PIT"] = tf.reshape(PIT [cl],[-1,40,40,1])
      
      for degrees in [0,45,90,135]: 
          predictions[cl+"_v1_"+str(degrees)] = v1[cl][degrees] #already reshaped      
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  


  predictions = {
      # Generate predictions (EVAL mode)
      "classes": tf.argmax(input=AIT, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(AIT, name="softmax_tensor")
  }




  




  # Calculate Loss (for both TRAIN and EVAL modes)
  one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth= 5)  
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels	, logits=AIT)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
  # change optimiser if wanted GradientDescent
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)# was 0.001 
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




