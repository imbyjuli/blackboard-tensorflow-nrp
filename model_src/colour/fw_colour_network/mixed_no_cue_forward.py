
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
from input_data import read_data_four_forms,read_data_four_forms_predict
from input_nrp_data import read_nrp_data, read_nrp_data_predict
from input_data_coordinates import read_data_four_forms_predict,extra_data

from forward_model_cue import ventral_feed_forward 
import tensorflow as tf






batch_size = 200





def main(unused_argv):

  #TODO: set with flag not manually
  nrp_data_used = False 



  if nrp_data_used: 
    data_set_directory = "../custom_dataset/nrp_set_bw/"
    model_dir = "/tmp/nrp_model"
    read_data = read_nrp_data
    read_data_predict = read_nrp_data_predict
  else:   
    data_set_directory1 = '../custom_dataset/bw_set/'
    data_set_directory = '../custom_dataset/4bw_set/'
    model_dir = "/tmp/no_cue_model"
    read_data = read_data_four_forms
    read_data_predict = read_data_four_forms_predict
  





  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=ventral_feed_forward, model_dir=model_dir)
  


  #'''
 # Set up logging for predictions
 # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}


  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  
  print("Reading Train data")
  
  train_data, train_cues, train_labels,_ = extra_data(data_set_directory1,"extra",True)  # Returns np.array

  '''
  train_data, train_cues, train_labels,_ = extra_data(data_set_directory1,"extra_fit",True)  # Returns np.array
  train_data1, train_cues1, train_labels1 = read_data(data_set_directory,"train",True)  # Returns np.array

  train_data = np.concatenate((train_data,train_data1),axis = 0)
  train_cues = np.concatenate((train_cues,train_cues1),axis = 0)
  train_labels = np.concatenate((train_labels,train_labels1),axis = 0)
  '''
  print("done")
  print(train_data.shape) 





  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data,"cues":train_cues},
      y=train_labels,
      batch_size= batch_size,
      num_epochs=None,
      shuffle=True)
  




  classifier.train(
      input_fn=train_input_fn,
      steps=5000,
      hooks=[logging_hook])
  #'''









  # read in eval data after done: 

  #'''
  eval_data, eval_cues, eval_labels,_ = extra_data(data_set_directory1,"extra_eval",True)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data,"cues":eval_cues},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)


  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  
  '''
  tensors_to_log_predict = {"probabilities": "softmax_tensor"}
    
  logging_hook_predict = tf.train.LoggingTensorHook(
      tensors=tensors_to_log_predict, every_n_iter=1)




  predict_data, predict_cues, predict_labels = read_data_predict(data_set_directory,"predict",True) # returns data, cues as well as location of the cue (not 5 arrays like in train and eval) 
  predict_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": predict_data,"cues": predict_cues},
      y=None,
      num_epochs=1,
      shuffle=False)
  predict_result = classifier.predict(input_fn=predict_fn, hooks = [logging_hook_predict])



  predict_data_2,predict_cues_2, predict_labels_2 = read_data_predict(data_set_directory,"predict_2",True)
  predict_fn_2 = tf.estimator.inputs.numpy_input_fn(
      x={"x": predict_data_2,"cues": predict_cues_2},
      y=None,
      num_epochs=1,
      shuffle=False)
  predict_result_2 = classifier.predict(input_fn=predict_fn_2, hooks = [logging_hook_predict])

  with open('predict_files.pkl', 'w') as f:
    pickle.dump([list(predict_result),predict_labels,list(predict_result_2),predict_labels_2], f)
    #'''



if __name__ == "__main__":
  tf.app.run()
