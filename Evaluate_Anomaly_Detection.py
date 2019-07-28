import os
import numpy as np
from scipy.io import loadmat, savemat
import cv2
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy.matlib


eval_workspace_path = '/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/'

# C3D features for videos (*.fc6-1)
C3D_CNN_Path = eval_workspace_path + 'Testing_Videos_C3D'
# Path of mp4 videos
Testing_VideoPath = eval_workspace_path + 'Testing_Videos'
# Path of Temporal Annotations 
AllAnn_Path = eval_workspace_path + 'Temporal_Annotations/Matlab_formate'
# Path of Pretrained Model score on Testing videos (32 numbers for 32 temporal segments)
Model_Score_Folder = eval_workspace_path + 'output_test_anomaly'
# Path to save results. 
Paper_Results='/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Paper_Results/'


All_Videos_scores = os.listdir(Model_Score_Folder)
nVideos = len(All_Videos_scores)
frm_counter = 1
All_Detect = np.zeros((1, 10000000))
All_GT = np.zeros((1, 10000000))


Testing_Videos1 = os.listdir(AllAnn_Path)


for ivideo in range(nVideos):
    #Access each annotation files for (each) videos
    Ann_Path = AllAnn_Path + '/' + All_Videos_scores[ivideo]
    single_vid_temporal_anno = loadmat(Ann_Path)
    
    #Check that each predicted score of test set contains corresponding annotation files
    if(All_Videos_scores[ivideo][:-4] != Testing_Videos1[ivideo][:-4]):
        print('Mismatch between predicted score and annotation files!!')
        
    VideoPath = Testing_VideoPath + '/' + All_Videos_scores[ivideo][:-6] + '.mp4' #Path to each video
    ScorePath = Model_Score_Folder + '/' + All_Videos_scores[ivideo][:-4] + '.mat' #Path to video's predicted score

    
    #Load the predictions for each video (32-dim)
    Predic_scores = loadmat(ScorePath) # Predic_scores['predictions']
    
    cap = cv2.VideoCapture(VideoPath)
    Actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Folder_Path = C3D_CNN_Path + '/' + All_Videos_scores[ivideo][:-6]
    All_Files = os.listdir(Folder_Path)
    nFileNumbers = len(All_Files)
    nFrames_C3D = nFileNumbers * 16 #Compute features for every 16 frames
    
    
    #### 32 shots ####
    
    Detection_score_32shots = np.zeros((1, nFrames_C3D))
    #Returns evenly spaced 32 shots based on total fc6-1 files for each video
    Thirty2_shots = np.linspace(start=1, stop=nFileNumbers, num=33)
    Thirty2_shots = Thirty2_shots.round(0).astype(np.int)
    p_c=0
    
    
    for ishots in range(Thirty2_shots.size - 1):
        ss = Thirty2_shots[ishots]
        ee = Thirty2_shots[ishots + 1] - 1
        
        if (ishots == len(Thirty2_shots)):
            ee=Thirty2_shots[ishots+1]
            
            
        if (ee < ss):
            Detection_score_32shots[:, (ss-1)*16:(ss-1)*16+1+15] = Predic_scores['predictions'][p_c] # every 16 items
        else:
            Detection_score_32shots[:, (ss-1)*16:(ee-1)*16+16] = Predic_scores['predictions'][p_c] # every 48 items
        p_c += 1
        
            
    Final_score = np.concatenate((Detection_score_32shots, 
                                 np.matlib.repmat(Detection_score_32shots[:, -1], 1, Actual_frames - Detection_score_32shots.shape[1])),
                                 axis=1)
    GT = np.zeros((1, Actual_frames))
        
        
    only_temporal_annotation = single_vid_temporal_anno['Annotation_file']['Anno']
    temp_ann = only_temporal_annotation[0,0]
    for i in temp_ann:
        start_frm = i[0]
        end_frm = i[1]
        GT[:, start_frm:end_frm] = 1 #Abnormal segments
   
        #For normal videos
        if start_frm == -1:
            GT = np.zeros((1, Actual_frames))
        
    

    #All_Detect stores score for every test video continuously
    All_Detect[:, frm_counter:frm_counter + Final_score.size] = Final_score 
    All_GT[:, frm_counter:frm_counter + Final_score.size] = GT
    frm_counter += Final_score.size
            

All_Detect = All_Detect[:, :frm_counter - 1]
All_GT = All_GT[:, :frm_counter - 1]            
labels = All_GT.flatten()
scores = All_Detect.flatten()




# Compute AUC curve
auc = roc_auc_score(labels, scores)
print('\nAUC score: %.4f' % auc)


fpr, tpr, _ = roc_curve(labels, scores)
matDict = {}
matDict['X'] = np.reshape(fpr, (fpr.size, 1))
matDict['Y'] = np.reshape(tpr, (tpr.size, 1))
matDict['AUC'] = auc

savemat(Paper_Results + 'testing.mat', matDict)
  
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
