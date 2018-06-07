from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import csv
from backward_weights import init_locally_backward_weights ,initiate_backward_weights
from backward_layers import ait_backwards,pit_backwards,v4_backwards,v2_backwards, create_direction_for_eval

tf.logging.set_verbosity(tf.logging.INFO)


def ventral_feed_backward(features, labels, mode):
#==========================================================================================
####### weights and biases described in v_layers_weights
#========================================================================================== 
  n_field_1 = 40 * 40 * 3                                
  use_sparse = True 

  input_layer_size = {
      #"classes": tf.argmax(input=logits, axis=1),
      "v2_1":       (40,40),
      "v2_2":         (40,40),
      "v4":         (40,40),
      "PIT":        (40,40),
      "AIT":         (5,1)
 }

  for degrees in [0,45,90,135]: 
      input_layer_size ["v1_"+str(degrees)] = (40,40)

  weights, bias = initiate_backward_weights(input_layer_size) 
  packed_backward_weights = init_locally_backward_weights(input_layer_size) 
#==========================================================================================
####### initiation
#==========================================================================================  
  


  input_layers = {}
  for key in input_layer_size.keys(): 
    img_x,img_y = input_layer_size [key]
    features_float  = tf.cast(features[key], tf.float32)
    input_layers[key]  = tf.reshape(features_float, [-1, img_x*img_y *   1]) 
  #print(input_layers.keys())


#==========================================================================================
####### layers described in v_layers
#========================================================================================== 
  AIT_b = ait_backwards(input_layers)
  PIT_b = pit_backwards(AIT_b,input_layers,packed_backward_weights, weights,bias, use_sparse = use_sparse) 
  v4_b =  v4_backwards(PIT_b,input_layers,packed_backward_weights,weights,bias,   use_sparse = use_sparse)
  v2_b,_,_ =  v2_backwards(v4_b,input_layers,packed_backward_weights,weights,bias, use_sparse = use_sparse)
  



  #  final_dense = tf.layers.dense(v2_backward)
  logits = v2_b#tf.layers.dense(inputs = v2_b, units = 40 * 40)
  tf.summary.image("AIT_b",tf.reshape(AIT_b,[-1,40,40,1]),1)
  tf.summary.image("PIT_b",tf.reshape(PIT_b,[-1,40,40,1]),1)
  tf.summary.image("v4_b",tf.reshape(v4_b,[-1,40,40,1]),1)
  tf.summary.image("v2_b",tf.reshape(v2_b,[-1,40,40,1]),1)
  



  #  final_dense = tf.layers.dense(v2_backward)
  logits = tf.layers.dense(inputs = v2_b, units = 4)
  tf.summary.image("logits",tf.reshape(logits,[-1,2,2,1]),1)



#==========================================================================================
####### Prediction with Tensorflow 
#========================================================================================== 



  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }



  if mode == tf.estimator.ModeKeys.PREDICT:

   #sess.run(print_activation_dict(return_tensors))  
    

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)



  # Calculate Loss (for both TRAIN and EVAL modes)
  one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)
  tf.summary.image("labels",tf.reshape(one_hot_labels,[-1,2,2,1]),1)  
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels  , logits=logits)

  # figure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
  # change optimiser if wanted 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
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
