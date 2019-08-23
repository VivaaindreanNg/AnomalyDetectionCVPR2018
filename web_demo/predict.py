from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import csv
import configparser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
import shutil

import theano.tensor as T
import theano



def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

def load_dataset_One_Video_Features(Test_Video_Path):

    VideoPath =Test_Video_Path
    f = open(VideoPath, "r")
    words = f.read().split()
    num_feat = int(len(words) / 4096)
    # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
    # we have already computed C3D features for the whole video and divided the video features into 32 segments.

    count = -1
    VideoFeatues = []
    for feat in range(0, num_feat):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            VideoFeatues = feat_row1
        if count > 0:
            VideoFeatues = np.vstack((VideoFeatues, feat_row1))
    AllFeatures = VideoFeatues

    return AllFeatures


def predict_frame_score(c3d_input_path):
    Model_dir='model/'
    weights_path = Model_dir + 'weightsAnomalyL1L2_20000.mat'
    model_path = Model_dir + 'model.json'
    model=load_model(model_path)
    load_weights(model, weights_path)

    inputs = load_dataset_One_Video_Features(c3d_input_path) 
    predicted_scores = model.predict_on_batch(inputs)  

    return predicted_scores
























