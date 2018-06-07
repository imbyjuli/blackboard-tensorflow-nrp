# Blackboard Architecture for Simulating Visual Search in the NRP: 

## Setup: 

1. Install NRP platform,following the instructions here: 
[https://bitbucket.org/hbpneurorobotics/neurorobotics-platform](https://bitbucket.org/hbpneurorobotics/neurorobotics-platform)  
2. paste the "traengle" folder into /Models/ 
3. add "traengle" line into the 
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