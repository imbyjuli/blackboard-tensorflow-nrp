# Imported Python Transfer Function
#
from gazebo_msgs.srv import SetModelState
import rospy
rospy.wait_for_service("/gazebo/set_model_state")
service_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistent=True)
@nrp.MapCSVRecorder("recorder", filename="test_labels.csv", headers=["name", "top_left","top_right","bottom_left","bottom_right","cl_tl","cl_tr","cl_bl","cl_br"])
@nrp.MapVariable("counter", initial_value = 0)
@nrp.MapVariable("top_right", initial_value={'x': 0.4, 'y': 1.38, 'z': 1.3})
@nrp.MapVariable("top_left", initial_value={'x': -0.4, 'y': 1.38, 'z': 1.3})
@nrp.MapVariable("bottom_right", initial_value={'x': 0.4, 'y': 1.38, 'z': 0.8})
@nrp.MapVariable("bottom_left", initial_value={'x': -0.4, 'y': 1.38, 'z': 0.8})
@nrp.MapVariable("last_positions", initial_value = [0,1,2,3] )
@nrp.MapVariable("last_color",     initial_value = [0,0,0,0] ) 
@nrp.MapVariable("last_t",     initial_value = 0.0 ) 
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
@nrp.MapVariable("phase", initial_value = None)
@nrp.MapVariable("bridge", initial_value = None)
@nrp.MapRobotSubscriber("camera", Topic('/icub_model/left_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.Robot2Neuron() # dummy R2N
def move_target(t, recorder, counter, top_right , top_left , bottom_right , bottom_left , last_positions, last_color, last_t, dropout,variance, camera, bridge, phase, set_model_state_srv):

    #TODO: change / create folder to store images in: (make sure the folder is created)
    dataset_dir = "/tmp/train_dataset/"
    #TODO: Change if you want to create black and white or coloured data
    use_colour = False 
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
    if(t > 10 and phase.value is None):
        phase.value = "SET"
#save data: 
    if (phase.value == "CAPTURE" and t > last_t.value +0.4): 
        from cv_bridge import CvBridge, CvBridgeError
        bridge.value = CvBridge()
    # no image received yet, do nothing
        if camera.value is None:
          return
    # convert the ROS image to an OpenCV image and Numpy array
        cv_image = bridge.value.imgmsg_to_cv2(camera.value, "rgb8")
        img = Image.fromarray(cv_image, 'RGB')
        tmp_filename = dataset_dir + 'data_' + str(counter.value) + '.png'
        filename = 'train_dataset/data_' + str(counter.value) + '.png'
        img.save(tmp_filename)  
        counter.value = counter.value + 1
        last_c = last_color.value
        var = variance.value 
        p = last_positions.value
        f = ["Circle","Diamond","Square","Triangle"]
        f = [f[i] if dropout.value[i] else "empty" for i in range(4)]
        sorted_forms = [f[p.index(i)] for i in range(4)]
        sorted_colours = [last_c[p.index(i)] for i in range(4)]
        sorted_colours = [colors[c] for c in sorted_colours] 
        recorder.record_entry(
            filename,
            sorted_forms[0],
            sorted_forms[1],
            sorted_forms[2],
            sorted_forms[3],
            sorted_colours[0],
            sorted_colours[1],
            sorted_colours[2],
            sorted_colours[3]
            )
        clientLogger.info(counter.value)
        phase.value = "SET"
        last_t.value = t 
    #random a new combination of forms as well as their pos variance and dropout    
    elif(phase.value == "SET" and t > last_t.value + 0.4): 
        random.shuffle(last_positions.value) 
        
        last_c =  random.sample(range(4),4) if use_colour else [3,3,3,3] 
        last_color.value = last_c
        dropout.value = [(random.random() > dropout_probability) for i in range(4)]
        phase.value = "CAPTURE"
        last_t.value = t
        var = [0.08 * random.random() -0.04 for i in range(8)] 
        variance.value = var 
    else: 
        last_c = last_color.value
        var = variance.value
    #this has to be re-done
    for i in range(4): 
        if dropout.value[i]:
            chosen_color =  colors[last_c[i]]
            position = positions [ last_positions.value[i]]
            chosen_form = forms[i]
            readjust = (0,0) if i<3 else (-0.1,0.15) 
            form_c_dict[chosen_form][chosen_color].pose.position.x = position["x"] + var[i] + readjust[0]
            form_c_dict[chosen_form][chosen_color].pose.position.y = position["y"] + readjust[1]
            form_c_dict[chosen_form][chosen_color].pose.position.z = position["z"] + var[i+2]
    #call service
    for form in forms: 
        for color in colors: 
            response = set_model_state_srv.value(form_c_dict[form][color])
          #check response        
            if not response.success: 
                clientLogger.info(response.status_message)
#
