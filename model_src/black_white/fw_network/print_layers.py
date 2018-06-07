
import scipy.misc
import numpy as np
import tensorflow as tf
import os

def print_activation_layer(layer, size, name):
	layer = layer.eval(sess.as_default()) 
	layer.reshape(size) 
	layer = layer + 1 
	layer = layer * (255 /2)  
	layer = layer.astype(int)	 
	print("saving image") 
	scipy.misc.imsave(name + ".png", layer)
  


def print_activation_dict (return_tensors): 
	#dict_keys = return_tensors.keys()
	

	for key,item_dict  in return_tensors.items() : 
		print_activation_layer(layer = item_dict["layer"],size = item_dict["size"]    , name = key)


#turns an array from [-1 to 1] to colour scheme for better representation: 
def tensor_to_colour(tens):
	
	#lam_function1 = lambda x: tf.maximum(0,tf.multiply(-255,tf.cast(x,tf.int32)))
	#lam_function2 = lambda x: tf.maximum(0,tf.multiply( 255,tf.cast(x,tf.int32)))
    
	tens1 =  tf.maximum(tf.cast(0. ,tf.float32) ,tf.multiply(tf.cast(-1,tf.float32) ,tens)) 
	tens2 =  tf.maximum(tf.cast(0.,tf.float32) ,tens) #tf.multiply(255.0  ,tens))
	tens3 = tf.zeros(tens.shape,tf.float32)
	full_tensor =  tf.concat([tens1,tens2,tens3], 3)
	return full_tensor#tf.cast(full_tensor, tf.uint8)



def save_images_from_event(fn, tag, output_dir='./'):
	
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)
    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1  


def save_np_from_event(fn, tag, output_dir='./'):
	
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)
    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    print(v) 

