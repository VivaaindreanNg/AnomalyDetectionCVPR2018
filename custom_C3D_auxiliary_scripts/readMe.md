1. Simply place both of these files in repo: C3D/C3D-v1.0/examples/c3d_feature_extraction directory:

...
└── c3d_feature_extraction
    ├── input
    │   ├── frm
    │   └── Videos (Here contains the entirety of video needed for this project)
    │       ├── Abuse
    │       ├── Arrest
    │       ├── Arson
    │       ├── Assault
    │       ├── Burglary
    |       └── ...
    ├── *anomaly_detection_video.sh*
    └── *create_clips_for_extract.py*



2. Run via command: 

```
python create_clips_for_extract.py
```
* Output: Automatically create clips and perform C3D extraction for every videos in the sub-directories of c3d_feature_extraction/input/Videos/

