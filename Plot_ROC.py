from scipy.io import loadmat, savemat
import os
from os import listdir
from matplotlib import pyplot as plt
import numpy as np

ROC_Paper_Results_path = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/Paper_Results"

roc_files = listdir(ROC_Paper_Results_path)
roc_files.sort()

num_roc_files = len(roc_files)

color_list = ['blue', 'cyan', 'black', 'red']
roc_category = ['Binary classifier', 'Lu et al.', 
                'Hassan et al.', 'Proposed with constraint'
                ]

for i in range(num_roc_files):
    path_each_Roc = os.path.join(ROC_Paper_Results_path, roc_files[i])
    obtain_result = loadmat(path_each_Roc)
    
    X_val = obtain_result['X']
    Y_val = obtain_result['Y']
    auc = np.reshape(obtain_result['AUC'], -1)
    auc *=100
    auc = "%.2f" % auc
    plt.plot(X_val, Y_val, color=color_list[i], 
             label='{} ({})'.format(roc_category[i], auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC comparison (AUC score)")
    plt.grid(True)
    plt.legend(loc="lower right")
plt.show()


