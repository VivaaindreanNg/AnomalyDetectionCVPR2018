import os
import subprocess


eval_workspace_path = '/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/'

#Path below points to a dir storing testing video files (mp4 file)
dest_vid_name = eval_workspace_path + "Testing_Videos/"

#Path below points to the dir containing all extracted C3D features (fc6-1 files)
src_path_c3d = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_extracted_features"

#Path below points to the dir storing only testing (fc6-1) files for evaluation
dest_path_c3d = eval_workspace_path + "Testing_Videos_C3D/"

#Path below points to the entirety of the input video
path_containing_videos = "/home/vivaainng/C3D/C3D-v1.0/examples/c3d_feature_extraction/input/Videos"

#Path containing temporal anomaly for each video (_C.mat file)
temporal_ann_path = eval_workspace_path + 'Temporal_Annotations/'

def get_testing_video():

    if not os.path.exists(dest_vid_name):
        os.makedirs(dest_vid_name)
    
    #Get current path
    curr_path = subprocess.getoutput("pwd")

    #Enter the subdir containing all the input videos
    os.chdir(path_containing_videos)

    list_of_test_files = os.listdir(temporal_ann_path)
    list_of_test_files.sort()
    for test_f in list_of_test_files:
        vid_name = test_f[:-6] + '.mp4'

        #Find path leading to each individual testing video (mp4 file)
        src_vid_name = subprocess.getoutput("find -name {}".format(vid_name))
        src_vid_name = src_vid_name.split('\n')
        src_vid_name = src_vid_name[0]

        #Copying the files
        os.system("cp {} {}".format(src_vid_name, dest_vid_name))
  
    #Return back to original path prior
    os.chdir(curr_path)


def get_testing_c3d_features():

    if not os.path.exists(dest_path_c3d):
        os.makedirs(dest_path_c3d)
    
    curr_path = subprocess.getoutput("pwd")
    os.chdir(src_path_c3d)

    list_of_test_files = os.listdir(temporal_ann_path)
    list_of_test_files.sort()
    for test_f in list_of_test_files:
        c3d_name = test_f[:-6]

        src_c3d_name = subprocess.getoutput("find -name {}".format(c3d_name))
        src_c3d_name = src_c3d_name.split('\n')
        src_c3d_name = src_c3d_name[0]
        
        os.system("cp -r {} {}".format(src_c3d_name, dest_path_c3d))

    os.chdir(curr_path)

if __name__ == '__main__':
    get_testing_video()
    get_testing_c3d_features()




