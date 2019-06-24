for full_path_vid_file in $(find -name "*.mp4" | sort)
do

#echo "${full_path_vid_file#./input/avi/}"
python create_clips_for_extract.py ${full_path_vid_file#./input/avi/} 16
# python create_clips_for_extract.py <crimetype/video.mp4> <frames_per_clip>

#Remove extension 
path_vid=${full_path_vid_file%.*} 

#Get type of dir (immediate parent folder, ie: Abuse, Explosion,...) only
mkdir -p /home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_extracted_features/${path_vid#./input/avi/}

GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt conv3d_deepnetA_sport1m_iter_1900000 -1 50 20 prototxt/output_list_video_prefix.txt fc6-1
done



#**********************************For single input video**********************************
#mkdir -p <path/to/output>/crimetype/video 

#Changed <num_mini_batches> from 1 to 20
#GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt conv3d_deepnetA_sport1m_iter_1900000 -1 50 20 prototxt/output_list_video_prefix.txt fc6-1

