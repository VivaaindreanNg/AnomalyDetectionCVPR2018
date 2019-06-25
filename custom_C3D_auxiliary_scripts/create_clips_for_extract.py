import sys
import cv2
import os.path

#python create_clips_for_extract.py <crimetype/video.mp4> <frames_per_clip>

prototxt_path = "/home/vivaainng/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/"

input_vid_path = "input/"
output_clip_path = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_extracted_features/"

_video_path_list = sys.argv[1] # <crimetype/video.mp4> ie: Abuse/Abuse001_x264.mp4, Explosion/Explosion...mp4

input_video_prefix = os.path.join(prototxt_path, "input_list_video.txt")

output_video_prefix = os.path.join(prototxt_path, "output_list_video_prefix.txt")

input_prefix_file = open(input_video_prefix, 'w')
output_prefix_file = open(output_video_prefix, 'w')

#for video_path in _video_path_list:
input_video_path = os.path.join(input_vid_path, _video_path_list)
output_path = os.path.join(output_clip_path, _video_path_list)
cap = cv2.VideoCapture(input_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Segregate video into clips with n consecutive frames 
# starting from frame 0th
frames_per_clip = sys.argv[2]
frames_per_clip = int(frames_per_clip)
clip = int(total_frames/frames_per_clip) 
final_clip = clip*frames_per_clip
for i in range(final_clip):
	if i%frames_per_clip == 0:
	    # For input_prefix file
	    input_prefix_file.write('{} {} {}\n'.format(input_video_path, i, 0))

	    # For output_prefix file
	    #TODO: Update TESTING.sh with c3d_sport1m...extraction.sh scripts
	    output_prefix_file.write('{}/{}\n'.format(os.path.splitext(output_path)[0], ("%06d" % i)))

input_prefix_file.close()
output_prefix_file.close()
