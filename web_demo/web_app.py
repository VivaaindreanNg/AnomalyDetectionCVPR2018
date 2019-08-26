from flask import Flask, request, render_template, url_for
from flask import  Response, redirect, make_response, jsonify
from scipy.io import loadmat
import io
import base64
import cv2
import time
import os
from werkzeug.utils import secure_filename
from predict import predict_frame_score
import numpy as np
import numpy.matlib
import scipy.signal

os.system('python -m webbrowser -t "http://localhost:5000/index_page" ')

UPLOAD_FOLDER = '/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Test_Folder'

#Path that points to the Testing Videos (.mp4 files)
testing_vid_path = '/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/Testing_Videos/'

#Path that points to the Temporal Annotations for each testing videos (.mat files)
temp_anno_path = '/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Eval_Res/Temporal_Annotations/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/index_page') # Main Page
def index_page():
    return render_template('index.html')


def generate_frame():
    fps = 30
    f = open(os.path.join('static/input_c3d/video_name.txt'))
    video_path = f.read()
    f.close()

    cap = cv2.VideoCapture(str(video_path))

    while True:
        _, frame = cap.read()

        
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.rectangle(
            frame,
            (100, 0),
            (190, 25),
            (0, 0, 0),
            cv2.FILLED
        )
        cv2.putText(
            frame, 
            str(cap.get(cv2.CAP_PROP_POS_FRAMES)), 
            (100, 17), 
            font, 
            0.7, 
            (255, 255, 255), 
            1)

        

        imgencode = cv2.imencode('.jpg', frame)[1]
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n' + imgencode.tostring() + b'\r\n')
        time.sleep(0.2 / fps)

    cap.release()



@app.route('/load_video')
def load_video():
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    x = None
    y = None
    video = ''
    filename = ''
    
    # Save C3D averaged features & video name in path below
    saved_input_path = 'static/input_c3d/' 

    # Upload C3D features from Test_Folder/
    if request.method == 'POST':
        file = request.files['c3d_file']
        filename = file.filename #C3D path
        file.save(os.path.join(saved_input_path, 'c3d_input_txt'))
        video = testing_vid_path + filename[:-6] + '.mp4'


        #--------Include the name of video file--------
        f = open(os.path.join(saved_input_path + 'video_name.txt'), 'w')
        f.write(testing_vid_path + filename[:-6] + '.mp4')
        f.close()

        
        cap = cv2.VideoCapture(video)
        max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        #---- Prediction of scores for each 32 segments ------
        c3d_input_path = saved_input_path + 'c3d_input_txt'
        predicted_scores = predict_frame_score(c3d_input_path)


        
        total_segments = np.linspace(1, max_frame, num=33)
        total_segments = total_segments.round()

        Frames_Score = []
        count = -1
        for iv in range(0, 32):
            F_Score = np.matlib.repmat(predicted_scores[iv],1,(int(total_segments[iv+1])-int(total_segments[iv])))
            count = count + 1
            if count == 0:
                Frames_Score = F_Score
            if count > 0:
                Frames_Score = np.hstack((Frames_Score, F_Score))

        x = np.linspace(1, max_frame, max_frame)
        scores = Frames_Score
        scores1 = scores.reshape((scores.shape[1],))
        y = scipy.signal.savgol_filter(scores1, 101, 3)
        x = x.tolist()
        y = y.tolist()

        # ----------Load temporal anomaly ----------
        temporal_ann_path = temp_anno_path + filename[:-3] + 'mat'
        temporal_ann = loadmat(temporal_ann_path)
        temporal_ann = temporal_ann['Annotation_file']['Anno']
        annotations = temporal_ann[0, 0]

        store_ann = []
        for i in annotations:
            start_frm = i[0]
            store_ann.append(start_frm)
            end_frm = i[1]
            store_ann.append((end_frm))

        annotation_list = []

        # For single temporal annotations
        if len(store_ann) == 2:
            for x_val in x:
                if x_val >= store_ann[0] and x_val <= store_ann[1]:
                    annotation_list.append(1)
                else:
                    annotation_list.append(0)

        # For two distinct temporal annotations 
        if len(store_ann) == 4:
            for x_val in x:
                if x_val >= store_ann[0] and x_val <= store_ann[1]:
                    annotation_list.append(1)              
                elif x_val >= store_ann[2] and x_val <= store_ann[3]:
                    annotation_list.append(1)
                else:
                    annotation_list.append(0)
        

    
    return render_template(
        'line_chart.html',
        values=y, 
        labels=x, 
        legend=filename[:-6], 
        annotations=annotation_list)



if __name__ == '__main__':
    app.run(debug=True)


'''
Input to the server:
1)Averaged C3D files with 32 segments with 4096-dim (.txt) via: ~/Test_Folder dir

2)Path to video files corresponding to extracted averaged C3D files, via: 
  ~/Eval_Res/Testing_Videos/ dir

3)Path to temporal annotations (.mat files) for each test set, via:
  ~/Eval_Res/Temporal_Annotations/ dir
'''