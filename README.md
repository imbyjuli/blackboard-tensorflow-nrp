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
### Train Forward Network 
After preparing the datasets, go into *model_src/WANTED_COLOUR_MODE/fw_network*. Here open the file *train_forward_network.py*. Edit the dir strings as described. Afterwards run the script by entering. Here you can aslo edit training parameters such as iterations and batch size. 

```
python train_forward_network.py
```
while in the directory. Now the network should start to train. 

###Train Backward Network 
After the forward network is done training, by opening the script *create_backward_dataset.py*, filling the asked for strings for the dataset and trained model directorys and after running in by entering 
```
python create_backward_dataset.py
```
in the console. 
This will create a pickle file in the parent directory, with predicted data and the layer output of each run. 
With this data the backward network is trained: 
1. Navigate to the backward network source folder *../bw_network*. 
2. now open the file *train_backward_network* and edit the asked for directory string. 
3. run the script 
```
python train_backward_network.py
```
###Use newly train Network. 
Edit the respective strings in the transfer function of the prediction experiment, to point towards your newly trained models. 


