import os
import re
import subprocess

#Path below points to txt file containing temporal annotations for testing videos
path_temporal_anomaly_ann = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/Temporal_Annotations/Temporal_Anomaly_Annotation.txt"

#Path below points to a dir storing testing video files (mp4 file)
dest_vid_name = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/Testing_Videos/"

#Path below points to the dir containing all extracted C3D features (fc6-1 files)
src_path_c3d = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_extracted_features"

#Path below points to the dir storing only testing (fc6-1) files for evaluation
dest_path_c3d = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/Testing_Videos_C3D/"

#Path below points to the entirety of the input video
path_containing_videos = "/home/vivaainng/C3D/C3D-v1.0/examples/c3d_feature_extraction/input/Videos"


def get_testing_video():
    #Get current path
    curr_path = subprocess.getoutput("pwd")

    #Enter the subdir containing all the input videos
    os.chdir(path_containing_videos)

    f = open(path_temporal_anomaly_ann, "r")
    line = f.readlines()
    for x in line:
        video_name = re.findall(r"\w+.mp4\b", x.rstrip()) #Get substring ends with .mp4
        vid_name = video_name[0] # video_name is of list type, make it to string type

        #Find path leading to each individual testing video (mp4 file)
        src_vid_name = subprocess.getoutput("find -name {}".format(vid_name))
        src_vid_name = src_vid_name.split('\n')
        src_vid_name = src_vid_name[0]

        #Copying the files
        os.system("cp {} {}".format(src_vid_name, dest_vid_name))
  
    #Return back to original path prior
    os.chdir(curr_path)


def get_testing_c3d_features():
    curr_path = subprocess.getoutput("pwd")

    os.chdir(src_path_c3d)
    f = open(path_temporal_anomaly_ann, "r")
    lines = f.readlines()
    for line in lines:
        c3d_name = re.findall(r"\w+\b", line.rstrip())
        c3d_file_name = c3d_name[0]

        src_c3d = subprocess.getoutput("find -name {}".format(c3d_file_name))
        src_c3d = src_c3d.split('\n')
        src_c3d = src_c3d[0]
        
        os.system("cp -r {} {}".format(src_c3d, dest_path_c3d))

    os.chdir(curr_path)


get_testing_video()
get_testing_c3d_features()




