# Imported Python Transfer Function
#
from gazebo_msgs.srv import SetModelState
import rospy
rospy.wait_for_service("/gazebo/set_model_state")
service_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistent=True)
@nrp.MapCSVRecorder("recorder", filename="test_labels.csv", headers=["name", "top_left","top_right","bottom_left","bottom_right"])
@nrp.MapVariable("counter", initial_value = 0)
@nrp.MapVariable("top_right", initial_value={'x': 0.4, 'y': 1.38, 'z': 1.3})
@nrp.MapVariable("top_left", initial_value={'x': -0.4, 'y': 1.38, 'z': 1.3})
@nrp.MapVariable("bottom_right", initial_value={'x': 0.4, 'y': 1.38, 'z': 0.8})
@nrp.MapVariable("bottom_left", initial_value={'x': -0.4, 'y': 1.38, 'z': 0.8})
@nrp.MapVariable("last_positions", initial_value = [0,1,2,3] )
@nrp.MapVariable("last_color",     initial_value = [0,0,0,0] ) 
@nrp.MapVariable("set_time",      scope = nrp.GLOBAL) 
@nrp.MapVariable("dropout",     initial_value = [True,True,True,True] ) 
#@nrp.MapVariable("scaling",  initial_vale = [1.0, 1.0, 1.0, 1.0] ) 
#variance: (position_number = 4 * (x,y,z) 3) list 
@nrp.MapVariable("variance",  initial_value = [  0.0, 0.0, 
                                                0.0, 0.0, 
                                                0.0, 0.0, 
                                                0.0, 0.0] ) 
#@nrp.MapVariable("tr_var", initial_value={'x': 0.0, 'y':0.0, 'z': 0.0})
#@nrp.MapVariable("tl_var", initial_value={'x': 0.0, 'y':0.0, 'z': 0.0})
#@nrp.MapVariable("br_var", initial_value={'x': 0.0, 'y':0.0, 'z': 0.0})
#@nrp.MapVariable("bl_var", initial_value={'x': 0.0, 'y':0.0, 'z': 0.0})
@nrp.MapVariable("set_model_state_srv", initial_value=service_proxy)
@nrp.MapVariable("phase", scope = nrp.GLOBAL)
@nrp.Robot2Neuron() # dummy R2N
def move_target(t, recorder, counter, top_right , top_left , bottom_right , bottom_left , last_positions, last_color, set_time, dropout,variance,phase, set_model_state_srv):
    import random 
    from PIL import Image
    dropout_probability = 0.25 
    forms = ["Circle","Diamond","Square","Triangle"]
    colors = ["Green","Red","Blue","White"] 
    top_right = top_right.value
    top_left = top_left.value
    bottom_right = bottom_right.value
    bottom_left = bottom_left.value
    positions = [top_left,top_right, bottom_left,bottom_right]
    #setting orientation since game 
    form_orientation_dict = {
    "Triangle": {"x":0.70710678118, "y":0, "z":0, "w":0.70710678118},
    "Circle" :  {"x":0.70710678118, "y":0, "z":0, "w":0.70710678118},
    "Square" :  {"x":0, "y":0, "z":0, "w":0}, 
    "Diamond":  {"x": 0, "y":0.38268343236, "z":0, "w":0.9}
    }
    form_c_dict = {
    "Triangle": {},
    "Circle" :  {},
    "Square" :  {}, 
    "Diamond":  {}
    }   
    for form in forms: 
        orientation = form_orientation_dict[form]
        for color in colors: 
            form_c_dict[form][color] = gazebo_msgs.msg.ModelState()
            form_c_dict[form][color].model_name = form + "_" + color
            form_c_dict[form][color].scale.x = form_c_dict[form][color].scale.y = form_c_dict[form][color].scale.z = 1.0
            # reference frame
            form_c_dict[form][color].reference_frame = 'world'
            #orientation
            form_c_dict[form][color].pose.orientation.x = orientation["x"]
            form_c_dict[form][color].pose.orientation.y = orientation["y"]
            form_c_dict[form][color].pose.orientation.z = orientation["z"]
            form_c_dict[form][color].pose.orientation.w = orientation["w"]
            #set default pose
            form_c_dict[form][color].pose.position.x = 0
            form_c_dict[form][color].pose.position.y = 4
            form_c_dict[form][color].pose.position.z = 0
    if(t > 5.0 and phase.value is None):
        phase.value = "SET"
    #random a new combination of forms as well as their pos variance and dropout    
    if(phase.value == "SET"):
        #UNCOMMENT FOR RANDOM CONSTELLATIONS ! 
        #random.shuffle(last_positions.value) 
        #last_c =  random.sample(range(4),4) 
        #last_color.value = last_c
        #dropout.value = [(random.random() > dropout_probability) for i in range(4)]      
        set_time.value = t
        var = [0.08 * random.random() -0.04 for i in range(8)] 
        variance.value = var
        phase.value = "SEARCHING"
    else: 
        last_c = last_color.value
        var = variance.value
    #this has to be re-done
    for i in range(4): 
        if dropout.value[i]:
            #TODO change the next line when different coloured shapes are desired
            chosen_color =  colors[3] # for bw only white [last_c[i]]
            position = positions [ last_positions.value[i]]
            chosen_form = forms[i]
            readjust = (0,0) if i<3 else (-0.1,0.15) 
            form_c_dict[chosen_form][chosen_color].pose.position.x = position["x"]  + readjust[0]#+ var[i]
            form_c_dict[chosen_form][chosen_color].pose.position.y = position["y"] + readjust[1]
            form_c_dict[chosen_form][chosen_color].pose.position.z = position["z"] #+ var[i+4]
    #call service
    for form in forms: 
        for color in colors: 
            response = set_model_state_srv.value(form_c_dict[form][color])
          #check response        
            if not response.success: 
                clientLogger.info(response.status_message)
#
