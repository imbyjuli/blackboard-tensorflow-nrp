from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import csv
import random 

#used for no opposition search 
form_dict = { "triangle":0,    "diamond": 1,    "square": 2,    "circle": 3}

#4_form_dict: different from form_dict by starting classes at 1 not 0 
four_form_dict={"triangle":1,    "diamond": 2,    "square": 3,    "circle": 4}



location_dict = {
        "top_left" : 0,
        "top_right" : 1,
        "bottom_left" : 2,
        "bottom_right" :3
        }

def reverse_dicts(dict): 
  reverse_dict = {value: key for key,value in dict.iteritems()}
  return reverse_dict

reverse_form      = reverse_dicts(form_dict)
reverse_location  = reverse_dicts(location_dict)





#=================================================================================================================================
#=========ONLY VERSION: several images with given location ======================================================================
#=================================================================================================================================


def read_data_four_forms (path, mode_name,black_white = False,num_classes = 4): 
  with open(path + mode_name + '_labels.csv','r') as csvfile:
      reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      csv_list =   [','.join(row).split(',') for row in reader][1:]

  #reading in images 
  if black_white: 
    data = [cv2.imread( (path + name), cv2.IMREAD_GRAYSCALE).flatten() for name,_,_,_,_ in csv_list]
  else: 
    data = [cv2.imread( path + name).flatten() for name,_,_,_,_ in csv_list]


  labels_temp = [[tl, tr, bl, br] for _,tl, tr, bl, br  in  csv_list]


  y_labels,cues = transform_temp_labels(labels_temp) 

  #transforming the name of the cue into an array of shape 4: with all zeros except one position: 
  #for example "triangle" -> [1. ,0 ,0 ,0]
  def make_ixd_array(cue_name):
      cue_array = np.zeros([num_classes],dtype = np.float32)
      cue_array[form_dict[cue]] = 1.
      return cue_array
  final_cues = [make_ixd_array(cue) for cue in  cues]

  #cast variables into format usable by input_fn of estimator 
  features = np.stack(data)  
  y_labels = np.asarray(y_labels)
  cue_feature = np.stack(final_cues) 
  return features, cue_feature, y_labels     







#helper function, used to turn "label" from csv (only the forms and theis position) into  a usable label for the algorithm 
#this is done by first taken a random form from the form_list and checking if this form is in the current image: 
#if YES: set class to class between 1 an 4 (could be expanded) 
#if NO: set class to 0 -> cued object not in image
#Returns two lists with the final label [0 - 4] and the given cue [1 - 4]
def transform_temp_labels(labels_temp): 
  form_list = ["triangle",    "diamond",    "square",    "circle"]
  
  def transform_label_row(row):   
    chosen_form = random.choice(form_list)
    if chosen_form in row: 
      return (four_form_dict[chosen_form], chosen_form)
    else:
      return (0, chosen_form)

  list_temp = [transform_label_row(label) for label in labels_temp]

  y_labels, cue = zip(* list_temp) 
  return y_labels,cue


#=================================================================================================================================
#=========READING DATA TO BE PREDICTED FOR FORWARD NETWORK AND USED IN BACKWARD NETWORK===========================================
#=================================================================================================================================





def read_data_four_forms_predict(path, mode_name,black_white = False,num_classes = 4): 
  with open(path + mode_name + '_labels.csv','r') as csvfile:
      reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      csv_list =   [','.join(row).split(',') for row in reader][1:]
  #remove "all empty" entries: 
  csv_list = [row for row in csv_list if any( location != "empty" for location in row[1:])]


  #reading in images 
  if black_white: 
    data = [cv2.imread( (path + name), cv2.IMREAD_GRAYSCALE).flatten() for name,_,_,_,_ in csv_list]
  else: 
    data = [cv2.imread( path + name).flatten() for name,_,_,_,_ in csv_list]


  labels_temp = [[("top_left",tl), ("top_right",tr),("bottom_left", bl), ("bottom_right",br)] for _,tl, tr, bl, br  in  csv_list]

  
  cues, location_labels  = create_predict_labels (labels_temp) 


  #transforming the name of the cue into an array of shape 4: with all zeros except one position: 
  #for example "triangle" -> [1. ,0 ,0 ,0]
  def make_ixd_array(cue_name):
      cue_array = np.zeros([num_classes],dtype = np.float32)
      cue_array[form_dict[cue]] = 1.
      return cue_array


  final_cues = [make_ixd_array(cue) for cue in  cues]
  location_labels = [location_dict[l] for l in location_labels]




  #cast variables into format usable by input_fn of estimator 
  features = np.stack(data)  
  location_labels = np.asarray(location_labels)
  cue_feature = np.stack(final_cues)   
  return features,   cue_feature , location_labels 

#helper function, used to turn "label" from csv (only the forms and their position) into  a usable label for the predict on forward
#converting the data to be used in the backward pass
#for now the predict set does not contain empty images, furthermore it does not choose a cue that is not in the image (since if image cue isnt available it answers differently)
#returns two list: 
#CUE: name of the random cue 
#location of the chosen cue
def create_predict_labels(labels_temp): 


  # assuming lable_temp is not empty   
  def transform_row(row): 
    form_list = filter (lambda x  : x[1] != "empty", row)
    if(len(form_list) == 0): print(row)   
    chosen_loc,chosen_form = random.choice(form_list)
    return (chosen_loc,chosen_form) 
  list_temp = [transform_row(label) for label in labels_temp]

  location_labels, cue = zip(* list_temp) 
  return cue,location_labels
#this function assumes that the input data is a list of dicts who all have the same keys 
#returns 

def convert_predicted_data(data): 
  keys = data[0].keys()
  return_dict = {}
  for key in keys:
    return_dict[key] = np.stack([datapoint_dict[key] for datapoint_dict in data]) 
  return return_dict



