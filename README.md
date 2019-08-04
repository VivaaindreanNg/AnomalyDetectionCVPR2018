DATASET:

The dataset can be also downloaded from the following link:
https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset


Below you can find Training/Testing Code for our anomaly Detection project which was published in Computer Vision and Pattern Recognition, CVPR 2018.

The implementation is tested using:
```
Keras version 1.1.0 or 1.0.7
Theano 1.0.2 or 1.0.3
Numpy 1.15.4
Python 3
Ubuntu 16.04
```

We used C3D-v1.0 (https://github.com/facebook/C3D) with default settings as a feature extractor.

Directory custom_C3D_auxiliary_scripts contains scripts that automates the feature extraction into corresponding directories 
based on the category of each video.

### Sequences of executing this Project

Following contains step by step sequences on how to get this project up and running:

Step 1: Perform C3D Feature extractions onto the entirety of the input videos.
Head over to the custom_C3D_auxiliary_scripts for more info.
* Output: C3D extracted features segregated accordingly based on category (Abuse, Arrest, Burglary, etc.) in blob (.fc6-1) format:
├── C3D_extracted_features
    ├── Abuse
    |   ├── Abuse001_x264
    │   ├── Abuse002_x264
    │   ├── Abuse003_x264
    ...
    ├── Arrest
    |   ├── Arrest001_x264
    │   ├── Arrest002_x264
    │   ├── Arrest003_x264
    ├── Arson
    ├── Assault
    ├── Burglary
    ├── ...


Step 2: Perform the averaging of the extracted C3D features into 32 segments, each with 4096 dimension of feature vector.

```
Run the Save_C3DFeatures_32Segments.m file 
```
* Output: Averaged C3D extracted features segregated accordingly based on category (Abuse, Arrest, Burglary, etc.) in txt format (per each video):
├── C3D_features_txt_Avg
    ├── Abuse
    |   ├── Abuse001_x264_C.txt
    │   ├── Abuse002_x264_C.txt
    │   ├── Abuse003_x264_C.txt
    ...
    ├── Arrest
    |   ├── Arrest001_x264_C.txt
    │   ├── Arrest002_x264_C.txt
    │   ├── Arrest003_x264_C.txt
    ...
    ├── Arson
    ├── Assault
    ├── Burglary
    ├── ...



Step 3: Based on the averaged C3D features, split it into train and test sets. Enter the Anomaly_Detection_splits directory and run the command: 

```
python extract_input_for_train_test.py
```
* Output: 2 directories will be made, Train_Folder and Test_Folder. The former will contain 2 sub-directories (totalling up to 1610  samples), while the latter will contain 290 samples:
    ...
    └── Train_Folder
           ├── Training_Abnormal_Videos_Anomaly (810 samples)
           └── Training_Normal_Videos_Anomaly (800 samples)
    ...
    └── Test_Folder (290 samples)



Step 4: Train the model by running the following command:

```
python TrainingAnomalyDetector_public.py
```
* Output: Saved the model & weights for every 1000 iterations.



Step 5: Test the trained model & weights via:

```
python Test_Anomaly_Detector_public.py
```
* Output: Will store the 32 predicted scores for those 290 samples in the Eval_Res/output_test_anomaly directory. 


Step 6: Prepare necessary folders & files for evaluation. Run command below for automating the preparation of Videos (.mp4) & and C3D_extracted_features files (.fc6-1)

```
python extract_files_for_eval.py
```
* Output: Automatically create 2 directories & populate them:
    ├── Testing_Videos (290 .mp4 files)
    └── Testing_Videos_C3D (290 directories)
        ├── Abuse028_x264
        ├── Abuse030_x264
        ├── Arrest001_x264
        ...



Step 7: Evaluate the test set via:

```
python Evaluate_Anomaly_Detection.py
```
* Output: Will save the evaluation data (AUC in particular) in the Paper_Results directory.


Step 8: Visualize evaluation data by:

```
python Plot_ROC.py
```
* Output: The AUC graph of evaluated results, in comparison with previous results based on the files in directory Paper_Results/


### Description of each module
* Anomaly_Detection_splits contains a Python script that will segregate inputs for training (Anomaly only, as Normal videos are already segregated during feature extraction)
and testing.

* Anomaly_Train.txt contains the video names for training anomaly detector

* /Eval_Res/Temporal_Annotations/ contains ground truth annotations of the testing dataset.

* The project page can be found at: http://crcv.ucf.edu/projects/real-world/



## Citation:
```
@InProceedings{Sultani_2018_CVPR,
author = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},
title = {Real-World Anomaly Detection in Surveillance Videos},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```