
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import argparse
import sys
import tempfile
import csv
import numpy as np 
from input_nrp_data import read_nrp_data, read_nrp_data_predict
from forward_model import ventral_feed_forward 
import tensorflow as tf






batch_size = 100





def main(unused_argv):

  #TODO set data_set_directory 
  data_set_directory = "" #"../custom_dataset/nrp_set_bw/"
  #change model dir or copy the finished models before shutting down! 
  model_dir = "/tmp/nrp_model"
  read_data = read_nrp_data
  read_data_predict = read_nrp_data_predict

  
  assert data_set_directory !="", "Define directory path for dataset" 



  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=ventral_feed_forward, model_dir=model_dir)
  


  #'''
 # Set up logging for predictions
 # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}

  #logging and printing to tensorboarde very 100 steps (and printing in console)
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  
  print("Reading Train data")
  train_data, train_cues, train_labels = read_data(data_set_directory, black_white = True, nm_start =0 , nm_end = 10000)

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
 









  # read in eval data after done: 

  eval_data, eval_cues, eval_labels = read_data(data_set_directory,black_white = True,  nm_start =10000 , nm_end = 20000) 

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data,"cues":eval_cues},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)


  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  
if __name__ == "__main__":
  tf.app.run()
