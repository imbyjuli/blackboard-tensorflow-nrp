
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
from input_data_coordinates import read_data_four_forms_predict,extra_data
from input_nrp_data import read_nrp_data_predict
from forward_model_cue import ventral_feed_forward 
import tensorflow as tf






def main(unused_argv):
  
  print ("HI!")
  nrp_data_used = False 


  if nrp_data_used: 
    data_set_directory = "../custom_dataset/nrp_set_bw/"
    model_dir = "/tmp/nrp_model"
    read_data_predict = read_nrp_data_predict
  else:   
    data_set_directory = '../custom_dataset/bw_set/'
    model_dir = "/tmp/test_model"

    read_data_predict = read_data_four_forms_predict

  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=ventral_feed_forward, model_dir=model_dir)
  

  eval_data, eval_cues, eval_labels, eval_location = extra_data(data_set_directory,"extra_fix",True)
  print(eval_cues.shape)
  print(eval_labels.shape)

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data,"cues":eval_cues},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)


  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)



if __name__ == "__main__":
  tf.app.run()