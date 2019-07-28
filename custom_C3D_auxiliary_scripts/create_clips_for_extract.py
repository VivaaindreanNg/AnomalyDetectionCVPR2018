import sys
import cv2
import os.path
import numpy as np
import subprocess
import shlex
import time

start = time.time()

vids = []
for root,dirs,files in os.walk("./"):
    vids.extend((os.path.join(root,f) for f in files if f.endswith(".mp4")))

vids.sort()
for v in vids:
    crimetype_videoname = os.path.relpath(v, './input/Videos')

    prototxt_path = "/home/vivaainng/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/" 
    input_vid_path = "input/Videos/"
    output_clip_path = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_extracted_features/"
    
    video_path_list = crimetype_videoname # <crimetype/video.mp4> ie: Abuse/Abuse001_x264.mp4, Explosion/Explosion025_x264.mp4, Training_Normal_Videos_Anomaly/Normal_Videos122_x264.mp4
    
    input_video_prefix = os.path.join(prototxt_path, "input_list_video.txt")
    
    output_video_prefix = os.path.join(prototxt_path, "output_list_video_prefix.txt")
    
    input_prefix_file = open(input_video_prefix, 'w')
    output_prefix_file = open(output_video_prefix, 'w')

	#for video_path in video_path_list:
    input_video_path = os.path.join(input_vid_path, video_path_list)
    output_path = os.path.join(output_clip_path, video_path_list)
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Segregate video into clips with n consecutive frames 
	# starting from frame 0th
    frames_per_clip = 16
    frames_per_clip = int(frames_per_clip)
    clip = int(total_frames/frames_per_clip)
    final_clip = clip*frames_per_clip
    for i in range(final_clip):
        if i%frames_per_clip == 0:
		    # For input_prefix file
            input_prefix_file.write('{} {} {}\n'.format(input_video_path, i, 0))

		    # For output_prefix file
            output_prefix_file.write('{}/{}\n'.format(os.path.splitext(output_path)[0], ("%06d" % i)))
                
    input_prefix_file.close()
    output_prefix_file.close()
    num_mini_batch = int(np.ceil((total_frames/16)/50))

    #Remove video's extension
    crimetype_videoname = crimetype_videoname.split('.')[0]

    #Ensure anomaly_detection_video.sh file is executable to avoid permission error
    os.system('sh anomaly_detection_video.sh {} {}'.format(crimetype_videoname, str(num_mini_batch)))


end = time.time() - start

print('Total time taken for feature extraction: {0:.3f} seconds'.format(end))