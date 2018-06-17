
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import argparse
import sys
import tempfile
import csv
import numpy as np 
from input_backward_data import  convert_predicted_data
from backward_model_direction import ventral_feed_backward 
import tensorflow as tf
import pickle 



tf.logging.set_verbosity(tf.logging.INFO)


batch_size = 100




def main(unused_argv):
  #TODO set model dir 
  model_dir = "/tmp/backward_model"
  assert model_dir != "", 
  # Create the Estimator
  classifier = tf.estimator.Estimator(model_fn=ventral_feed_backward, model_dir=model_dir)

  #'''
  tensors_to_log = {"probabilities": "softmax_tensor"}
  

  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

  print("Reading Train data")
  with open('../predict_files.pkl', 'r') as f: 
    train_data, train_labels, _,_= pickle.load(f)



  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x= convert_predicted_data(train_data),
      y= train_labels,
      batch_size= batch_size,
      num_epochs=None,
      shuffle=True)
  
  classifier.train(
      input_fn=train_input_fn,
      steps=40000,
      hooks=[logging_hook])
  #'''









# read in eval data after done: 

  with open('../predict_files.pkl', 'r') as f: 
    _,_,eval_data,eval_labels= pickle.load(f)
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x= convert_predicted_data(eval_data),
      y= eval_labels,
      num_epochs=1,
      shuffle=False)


  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
