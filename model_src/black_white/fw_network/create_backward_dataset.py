
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import argparse
import sys
import tempfile
import csv
import numpy as np 
import pickle
from input_nrp_data import read_nrp_data_predict
from forward_model import ventral_feed_forward 
import tensorflow as tf






def main(unused_argv):

  #TODO set the data_directory as well as the model_directory 
  #data_directory can be the same as the training/eval directory
  data_set_directory = ""
  model_dir = ""
  read_data_predict = read_nrp_data_predict
  
  assert data_set_directory !="", "Define directory path for dataset" 
  assert model_dir !="", "Define directory path for trained model" 



  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=ventral_feed_forward, model_dir=model_dir)
  

  
  tensors_to_log_predict = {"probabilities": "softmax_tensor"}
    
  logging_hook_predict = tf.train.LoggingTensorHook(
      tensors=tensors_to_log_predict, every_n_iter=1)




  predict_data, predict_cues, predict_labels = read_data_predict(data_set_directory,black_white = True,nm_start =0 , nm_end = 10000) # returns data, cues as well as location of the cue (not 5 arrays like in train and eval) 
  predict_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": predict_data,"cues": predict_cues},
      y=None,
      num_epochs=1,
      shuffle=False)
  predict_result = classifier.predict(input_fn=predict_fn, hooks = [logging_hook_predict])
  #predict_result = (list(predict_result))


  predict_data_2,predict_cues_2, predict_labels_2 = read_data_predict(data_set_directory,black_white = True,nm_start =10000 , nm_end = 20000)
  predict_fn_2 = tf.estimator.inputs.numpy_input_fn(
      x={"x": predict_data_2,"cues": predict_cues_2},
      y=None,
      num_epochs=1,
      shuffle=False)
  predict_result_2 = classifier.predict(input_fn=predict_fn_2, hooks = [logging_hook_predict])

  with open('../data_for_backward_training.pkl', 'w') as f:
    pickle.dump([list(predict_result),predict_labels,list(predict_result_2),predict_labels_2], f)
    #'''



if __name__ == "__main__":
  tf.app.run()
