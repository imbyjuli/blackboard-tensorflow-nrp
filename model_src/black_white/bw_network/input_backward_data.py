import numpy as np 






#this function assumes that the input data is a list of dicts who all have the same keys 
#returns 

def convert_predicted_data(data): 
  keys = data[0].keys()
  return_dict = {}
  for key in keys:
    return_dict[key] = np.stack([datapoint_dict[key] for datapoint_dict in data]) 
  return return_dict