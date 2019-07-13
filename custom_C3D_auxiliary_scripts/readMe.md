1. Simply place both of these files in repo: C3D/C3D-v1.0/examples/c3d_feature_extraction directory.

2. Run sh c3d_sport1m_feature_extraction_video.sh to extract features of **every video** that is in 
the ../c3d_feature_extraction/input/Videos directory.

3. To only extract features for a **single** video:
(i)Run python create_clips_for_extract.py <crimetype/video.mp4> <num_frames_per_clip> to auto-generate 
auxiliary files(for both input & output) in c3d_feature_extraction/prototxt/ directory. (default num_frames_per_clip = 16, **number of mini-batches has increased from 1 to 40**)

(ii)Next, in c3d_sport1m_feature_extraction_video.sh, perform some following changes:
 - comment the entire for loop 
 - Uncomment the line mkdir -p <path/to/output>/crimetype/video , and change it corresponding to your <crimetype/video> that you've already select as input
 - Run c3d_sport1m_feature_extraction_video.sh

