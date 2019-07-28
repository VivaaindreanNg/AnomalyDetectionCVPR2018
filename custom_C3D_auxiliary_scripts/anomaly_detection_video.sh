mkdir -p /home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_extracted_features/$1

GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt conv3d_deepnetA_sport1m_iter_1900000 0 50 $2 prototxt/output_list_video_prefix.txt fc6-1


