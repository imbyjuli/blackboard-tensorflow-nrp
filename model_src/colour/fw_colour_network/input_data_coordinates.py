from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import csv
import random 
import scipy.stats as st


#used for no opposition search 
form_dict = { "triangle":0,    "diamond": 1,    "square": 2,    "circle": 3}

#4_form_dict: different from form_dict by starting classes at 1 not 0 
four_form_dict={"triangle":1,    "diamond": 2,    "square": 3,    "circle": 4}





def reverse_dicts(dict): 
  reverse_dict = {value: key for key,value in dict.iteritems()}
  return reverse_dict

reverse_form      = reverse_dicts(form_dict)



#=================================================================================================================================
#=========READING DATA TO BE PREDICTED FOR FORWARD NETWORK AND USED IN BACKWARD NETWORK===========================================
#=================================================================================================================================


#since the reading from the csv files is messed up this function helps filter out the tuple
def tuple_row(row): 
  #print(row)
  row = filter(lambda x: x !='',row)
  #print(row)
  #filterout empty entries

  row_int = [int(filter(str.isdigit, s)) for s in row]

  return [(row_int[i],row_int[i+1]) for i in range(0,len(row_int)-1,2)]





def read_data_four_forms_predict(path, mode_name,black_white = False,num_classes = 4): 
  with open(path + mode_name + '_labels.csv','r') as csvfile:
      reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      csv_list =   [','.join(row).split(',') for row in reader][1:]
  

  #remove "all empty" entries: 
  csv_list = [row for row in csv_list if any( location is not "empty" for location in row[1:5])]
  csv_list = [row[0:5] + tuple_row(row[5:]) for row in csv_list]

  print (mode_name)
  print (len(csv_list[1]))
  #reading in images 
  if black_white: 
    data = [cv2.imread( (path + row[0]), cv2.IMREAD_GRAYSCALE).flatten() for row in csv_list]
  else: 
    data = [cv2.imread( path + row[0]).flatten() for row in csv_list]

  labels_temp = [[(coords_tl,tl), (coords_tr,tr),(coords_bl, bl), (coords_br,br)] for _,tl, tr, bl, br,coords_tl, coords_tr, coords_bl, coords_br  in  csv_list]

  
  cues, location_labels  = create_predict_labels (labels_temp) 


  #transforming the name of the cue into an array of shape 4: with all zeros except one position: 
  #for example "triangle" -> [1. ,0 ,0 ,0]
  def make_ixd_array(cue_name):
      cue_array = np.zeros([num_classes],dtype = np.float32)
      cue_array[form_dict[cue]] = 1.
      return cue_array


  final_cues = [make_ixd_array(cue) for cue in  cues]
  




  #cast variables into format usable by input_fn of estimator 
  features = np.stack(data)  
  location_labels = np.asarray(location_labels)
  cue_feature = np.stack(final_cues) 
  
  return features,   cue_feature , location_labels 





#helper function, used to turn "label" from csv (only the forms and theis position) into  a usable label for the predict on forward
#converting the data to be used in the backward pass
#for now the predict set does not contain empty images, furthermore it does not choose a cue that is not in the image (since if image cue isnt available it answers differently)
#returns two list: 
#CUE: name of the random cue 
#location of the chosen cue


def create_predict_labels(labels_temp): 
  # assuming lable_temp is not empty
   
  def transform_row(row): 
    form_list = filter (lambda x  : x[1] != "empty", row)
    if form_list == []: print(row)
    chosen_loc,chosen_form = random.choice(form_list)
    return (chosen_loc,chosen_form) 
  list_temp = [transform_row(label) for label in labels_temp]

  location_coord, cue = zip(* list_temp) 
  return cue,location_coord





#=================================================================================================================================
#=========PLACING A GAUSS KERNEL ON TOP OF AN EXACT POSITION (RELAXING THE LOSS FUNCTION)=========================================
#=================================================================================================================================


#function placing a gauss bubble ontop of an exact 
def convert_positions(labels, outputshape = (40,40) ,kernel_size = 10): 
  
  return np.apply_along_axis(lambda x: kernel_in_img(outputshape,kernel_size ,x) ,1,labels)







# taken from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel



def kernel_in_img(img_size, kernel_size, location):
    background = np.zeros(img_size)
    x,y = location
    x_len, y_len = img_size
    #original img 240x240 -> 40x40 of final layer
    x = int(x_len * x / 240) 
    y = int(y_len * y / 240)
    kernel = gkern(kernlen = kernel_size, nsig = 3)
    kernel = kernel /kernel.max()
    
    radius = int (kernel_size / 2)
    radius_p = kernel_size -  radius #in case kernel_size % 2 == 1 
    
    x_start = max(0,x - radius)
    x_end  = min (x_len, x + radius_p) 
    xk_start = max(0, -(x-radius)) 
    xk_end = kernel_size + min (0, x_len - (x+radius_p)) 
    
    y_start = max(0,y - radius)
    y_end  = min (y_len, y + radius_p) 
    yk_start = max(0, -(y-radius)) 
    yk_end = kernel_size + min (0, y_len - (y+radius_p))   
    #print (x_start,x_end,xk_start,xk_end)

    
    
    background[x_start:x_end, y_start:y_end] = kernel[xk_start:xk_end, yk_start: yk_end]
    
    return background.flatten() 
    

    


#this function assumes that the input data is a list of dicts who all have the same keys 
#returns 

def convert_predicted_data(data): 
  keys = data[0].keys()
  return_dict = {}
  for key in keys:
    return_dict[key] = np.stack([datapoint_dict[key] for datapoint_dict in data]) 
  return return_dict






#=================================================================================================================================
#=========EXTRA FOR MIXED UP BUT SINGLE OBJECTS (also try with 0 cue)=============================================================
#=================================================================================================================================







def extra_data (path, mode_name,black_white = False,num_classes = 4): 

  labels_dict={"triangle":1,    "diamond": 2,    "square": 3,    "circle": 4}


  with open(path + mode_name + '_labels.csv','r') as csvfile:
      reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      csv_list =   [','.join(row).split(',') for row in reader][1:]


  csv_list = [row[0:5] + tuple_row(row[5:]) for row in csv_list]
  labels_temp = [[(coords_tl,tl), (coords_tr,tr),(coords_bl, bl), (coords_br,br)] for _,tl, tr, bl, br,coords_tl, coords_tr, coords_bl, coords_br  in  csv_list]
  #reading in images 
  if black_white: 
    data = [cv2.imread( (path + row[0]), cv2.IMREAD_GRAYSCALE).flatten() for row in csv_list]
  else: 
    data = [cv2.imread( path + row[0]).flatten() for row in csv_list]   
  labels_temp = [[(coords_tl,tl), (coords_tr,tr),(coords_bl, bl), (coords_br,br)] for _,tl, tr, bl, br,coords_tl, coords_tr, coords_bl, coords_br  in  csv_list]












  cues, location_labels  = create_predict_labels (labels_temp) 

  #y_labels,cues = transform_temp_labels(labels_temp) 

  #transforming the name of the cue into an array of shape 4: with all zeros except one position: 
  #for example "triangle" -> [1. ,0 ,0 ,0]
  def make_ixd_array(cue_name):
      cue_array = np.zeros([num_classes],dtype = np.float32)
      #cue_array[form_dict[cue]] = 1.
      return cue_array
  final_cues = [make_ixd_array(cue) for cue in  cues]

  #cast variables into format usable by input_fn of estimator 
  features = np.stack(data)  
  cue_feature = np.stack(final_cues)
  print(cues)
  y_labels = [labels_dict[cue] for cue in cues]
  y_labels = np.asarray(y_labels)
  
  #cue_feature = np.zeros_like(cue_feature)


  return features, cue_feature, y_labels,location_labels    





