# Blackboard Architecture for Simulating Visual Search in the NRP: 

## Setup: 

1. Install NRP platform,following the instructions here: 
[https://bitbucket.org/hbpneurorobotics/neurorobotics-platform](https://bitbucket.org/hbpneurorobotics/neurorobotics-platform)  
2. paste the "traengle" folder into /Models/ 
3. add "traengle" line into the textfile 
```
Models/_rpmbuild/models.txt
```
4. Recompile the NRP platform using: 
```
cd $HBP/user-scripts
./update_nrp build all
```
In case of problems, follow the instructions [here](https://bitbucket.org/hbpneurorobotics/neurorobotics-platform). If further problems occur consult the dedicated [forum](https://forum.humanbrainproject.eu/c/neurorobotics).
5. Install Tensorflow following the instructions in [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/) 
To note is that for the program is currently using the python2.7 version. Furthermore during development, Tensorflow version 1.5 of Tensorflow was used. 
6. Copy the content of *nrp_experiments* into *$NRP/Experiments* 
7. Open the experiments in the NRP environment and follow the instructions in the transfer functions.

## Create Dataset 
In case a new dataset is required for testing follow the following steps. 

1. Open the "Create Dataset"-Experiment. For this, two different parameters have to be set.  
    * dataset_dir:  the directory the test images will be saved in. Make sure the folder exists before running the experiment.  

    * use_colour: Determines wether or not Black and White data is created (only using white shapes) or coloured (white, green, blue, red).  Default set to False. 

2. After setting these variables, run the experiment. 
The experiment will automatically take pictures of the screen with random shape combinations, save the image and write the content and name of the image into a csv file. 
3. Extract this file go into the *.opt/nrpStorage/blackboard_create_dataset* directory. Here a folder named *csv_records_* followed by the date of creation, which contains the desired file *labels.csv*. 

4. Copy both the folder containing and the labels csv-file into a separate folder. Rename the folder containing the images to "train_dataset". 

5. Preprocessing: to get the data ready for training the network, run the jupyter notebook *jupyter_notebooks/preprocessing.ipynb*, after following the instruction given in the cells.  
This may take a while. This step removes noise from the train images (all pixel values below a certain threshold). If this is not desired, set the threshold in function *highpass_filter* to 0. 

## Train Network



