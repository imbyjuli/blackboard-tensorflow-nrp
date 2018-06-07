import numpy as np
import tensorflow as tf
import itertools 

def get_indices (input_shape = 40 * 40, row_length = 40, k_size = 3): 
  rd = range(k_size)
  def make_row(row_i): 
    it_mask_pos = (list(itertools.product(rd,rd)))
    return  [[row_i, row_i - r - row_length * d]  for (r,d) in  it_mask_pos if row_i-r-row_length*d >=0  and (row_i % row_length) - r >= 0  ] 
  indicies = [make_row(i)  for i in range(input_shape)]

  #flatten index list 
  indicies_2 = [item for sublist in indicies for item in sublist]
  return indicies_2


def init_locally_backward_weights(input_layers_size,custom_s = 0.1):
  weights ={}
  #bias = {}
  indexes = {}
  for  name, size in input_layers_size.items(): 
    x, y = size
    id_x = get_indices (input_shape = x * y, row_length = x, k_size = 3)
    weights[name] = tf.Variable(tf.random_normal(shape = [len(id_x)], stddev = custom_s)) 
    #bias [name] = tf.Variable(tf.random_normal(shape = [x * y]))
    indexes[name] = id_x 
  return weights,  indexes   



def initiate_backward_weights (input_layers_size):
  weights ={}
  bias = {}
  for  name, size in input_layers_size.items(): 
    x, y = size
    weights[name] = tf.Variable(tf.random_normal(shape = [x * y])) 
    bias [name] = tf.Variable(tf.random_normal(shape = [x * y] ))
  return weights, bias 
