from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2




import csv

#from print_layers import print_activation_dict,tensor_to_colour
from cue_weights import  init_all_weights
from layers import v1_layer, v2_layer_cue, v4_layer_cue, PIT_layer_cue #, AIT_layer
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def ventral_feed_forward(features, labels, mode):
#==========================================================================================
####### weights and biases described in v_layers_weights
#========================================================================================== 
    # 40 x 40 neurons, 1 / 3 channels (rgb )

  n_field_1 = 40 * 40 

  #weights,_ = initiate_weights(n_field_1, n_field_2)
  #rf_weights,bias = initiate_rf_weights (n_field_1)
  
  packed_weights = init_all_weights(n_field_1)
#==========================================================================================
####### initiation
#========================================================================================== 

  image_size = [240,240]
  img_x,img_y =  image_size 
  features_float  = tf.cast(features["x"], tf.float32)
  input_layer = tf.reshape(features_float, [-1, img_x,img_y , 1]) #almost unneccessary 
  cues = features ["cues"]
  


#==========================================================================================
####### layers described in v_layers
#========================================================================================== 
  
  v1 = v1_layer(input_layer)
  v2_1,v2_2 = v2_layer_cue(v1,cues, packed_weights) 
  v4  = v4_layer_cue(v2_1,v2_2, cues, packed_weights)
  #pit = v4
  pit = PIT_layer_cue(v4,cues, packed_weights)

  #units = num_classes +  1
  
  logits = tf.layers.dense(inputs=tf.reshape(pit,[-1,40*40*1]), units= 5)
  #print(tf.shape(logits)) 












#==========================================================================================
####### print data to board: 
#========================================================================================== 

  #'''
  if mode == tf.estimator.ModeKeys.TRAIN:
    output_num = 1 
    for degrees in [0,45,90,135]:
      tf.summary.image("v1_"+str(degrees), tensor_to_colour( tf.reshape(v1[degrees],[-1,40,40,1])), output_num)

    tf.summary.image("v2_1",  tensor_to_colour(tf.reshape(v2_1,[-1,40,40,1])), output_num)
    tf.summary.image("v2_2",  tensor_to_colour(tf.reshape(v2_2,[-1,40,40,1])), output_num)
    tf.summary.image("v4",  tensor_to_colour(tf.reshape(v4,[-1,40,40,1])), output_num)    
    tf.summary.image("pit", tensor_to_colour(tf.reshape(pit,[-1,40,40,1])   ), output_num)
    tf.summary.image("cue",tf.reshape(cues,[-1,2,2,1]),output_num)
    #tf.summary.tensor_summary("ait", logits) 
    #for name,value in rf_weights.items(): 
      #tf.summary.image(name,  tensor_to_colour(tf.reshape(value,[-1,40*40,40*40,1])), output_num)    
  #'''


#==========================================================================================
####### Prediction with Tensorflow 
#========================================================================================== 







  if mode == tf.estimator.ModeKeys.PREDICT:

    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "v2_1":     tf.reshape(v2_1,[-1,40,40,1]),
      "v2_2":  tf.reshape(v2_2,[-1,40,40,1]),
      "v4":  tf.reshape(v4,[-1,40,40,1]),
      "PIT": tf.reshape(pit,[-1,40,40,1]),
      "AIT": logits, 
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    for degrees in [0,45,90,135]: 
      predictions["v1_"+str(degrees)] = v1[degrees] #already reshaped
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  


  predictions = {
      # Generate predictions (EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }




  




  # Calculate Loss (for both TRAIN and EVAL modes)
  one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth= 5)  
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels	, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
  # change optimiser if wanted GradientDescent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)# was 0.001 
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




