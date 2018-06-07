# Imported Python Transfer Function
@nrp.MapRobotSubscriber("camera", Topic('/icub_model/left_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapVariable("coordinates",initial_value = (160.,120.) , scope = nrp.GLOBAL) 
@nrp.MapVariable("bridge", initial_value=None)
@nrp.MapVariable("set_time", initial_value = 0.0, scope = nrp.GLOBAL)
@nrp.MapVariable("phase", initial_value = "SET", scope = nrp.GLOBAL) 
@nrp.MapVariable("given_cue", initial_value =0)
@nrp.Robot2Neuron()
def check_screen(t,camera, set_time, phase, bridge, coordinates,given_cue):

    
    if phase.value == "PROCESSING": 
        if t - set_time.value > 5.0:
            
            clientLogger.info("setting")
            coordinates.value =  (160.,120.)
            phase.value =  "SET"
            return 
    if (phase.value is not  "SEARCHING" and phase.value is not "CUE") or (t - set_time.value) < 1.: 
        return 
    clientLogger.info("looking at screen") 
	#Preparing Cue. In this version the "visual cue" is skipped and the experiment is rotating through all objects. 
    
    #"triangle",    "diamond",    "square",    "circle"
    cue = given_cue.value
    
    cue_arr =np.array([[0.,0.,0.,0.]],np.float32) 
    cue_arr[0][cue] = 1. 
    given_cue.value = (given_cue.value + 1) % 4 
    form_list=["Nothing", "triangle",    "diamond",    "square",    "circle"]
    clientLogger.info("Looking for: "+ form_list[cue+1])
    
    '''
    if phase == "CUE": 
        cue_arr = np.array([[0.,0.,0.,0.]],np.float32) 

    ''' 
    
    
    #TODO change for absolute path of "models_src/" folder 
    directory_path = ""
    #TODO: set to either "black_white" or "colour"
    mode = ""

    

    #TODO: path to pretrained_models or self trained model path. 
    #either case it should point towards a directory containing both forward and backward models named "forward_trained" and "backward_trained"
    trained_model_path = directory_path +"../" + mode + "/"


    directory_path += mode +"/"


    #log the first timestep (20ms), each couple of seconds
    # import TensorFlow in the NRP, update this path for your local installation
    try:
        import site
        #site.addsitedir('<path to tensorflow venv>/lib/python2.7/site-packages')
        import tensorflow as tf
        import sys 
    	sys.path.insert(0,directory_path+ "fw_network/")
    	sys.path.insert(0, directory_path+ "bw_network/") 
        from forward_model import ventral_feed_forward 
        from backward_model_direction import ventral_feed_backward
    except:
        clientLogger.info("Unable to import TensorFlow or model, check directory path")
        raise

    #Forward Pass for both cue and search phase 
    fw_model_dir = trained_model_path + "forward_trained/"
    forward_estimator =  tf.estimator.Estimator(model_fn = ventral_feed_forward,  model_dir = fw_model_dir)
    


    #take image and preprocess! 
    from cv_bridge import CvBridge, CvBridgeError   
    bridge.value = CvBridge()
    # no image received yet, do nothing\
    if camera.value is None:
        return
    # convert the ROS image to an OpenCV image and Numpy array\
    cv_image = bridge.value.imgmsg_to_cv2(camera.value, "mono8")
    def highpass_filter(image, threshold = 100): 
         #make sure both forms are working 
        #assert if input & background are same size
        #only works for greyscale for now
        ri = image [:,50:290]
        shp = ri.shape
        for i in range(shp[0]): 
            for j in range(shp[1]): 
                value = ri[i,j]
                if value < threshold: 
                    ri[i,j] = 0
        return ri
    image = highpass_filter(cv_image) 
    image = image.reshape([1,-1])

    
    forward_fn = tf.estimator.inputs.numpy_input_fn(
      x= {"x": image, "cues":cue_arr},  # [1,40,40,1] 
      y= None,
      batch_size= 1,
      num_epochs= 1,
      shuffle=False)
    forward_output = list(forward_estimator.predict(input_fn=forward_fn)) [0]
    
    found_class = forward_output["classes"]
    clientLogger.info("Found: " + form_list[found_class])
    
    
    
    if found_class is 0: 
        clientLogger.info("Did not find object")
        phase.value = "PROCESSING"
        return 
    
    
    if phase == "CUE": 
        cue = np.zero([4],np.float32)
        cue[forward_output["classes"]] = 1. 
        given_cue.value = cue
        phase.value = "SET_SEARCHING" 
        return 
    
    
    #SEARCH MODE FROM HERE ON ! 
    transformed_fw_output = {}
    for key in forward_output.keys():
        transformed_fw_output[key] = np.stack([forward_output[key]]) 
    bckw_model_dir = trained_model_path + "backward_trained/"
    backward_estimator = tf.estimator.Estimator(model_fn = ventral_feed_backward, model_dir = bckw_model_dir)
    backward_fn = tf.estimator.inputs.numpy_input_fn(
      x = transformed_fw_output, #output of forward layers
      y = None,
      batch_size = 1,
      num_epochs = 1,
      shuffle = False)
    backward_output = list(backward_estimator.predict(input_fn = backward_fn)) [0]

    location_list = ["top_left" , "top_right" ,   "bottom_left" ,        "bottom_right"]
    clientLogger.info("At position: " +  location_list [backward_output["classes"]])
    
    
    location_dict = [(130,80),(250,80.) , (130.,200.) , (250.,200.)]
    coordinates.value = location_dict[backward_output["classes"]]
    clientLogger.info("Switching to processing") 

    phase.value = "PROCESSING"
    set_time.value = t 
    
