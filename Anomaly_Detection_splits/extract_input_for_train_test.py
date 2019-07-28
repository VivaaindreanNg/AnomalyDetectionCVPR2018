import os
import subprocess

def main():
	curr_path = subprocess.getoutput("pwd")
	c3d_avg_txt_path = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/C3D_features_txt_Avg"

	anomaly_workspace_path = "/home/vivaainng/Desktop/AnomalyDetectionCVPR2018/"
	train_abnormal_folder_path = anomaly_workspace_path + "Train_Folder/Training_Abnormal_Videos_Anomaly/"
	test_folder_path = anomaly_workspace_path + 'Test_Folder/'

	#Create Train_Folder & Test_Folder dir
	if not os.path.exists(train_abnormal_folder_path):
		os.makedirs(train_abnormal_folder_path)
	if not os.path.exists(test_folder_path):
		os.makedirs(test_folder_path)

	#Extracting all Normal training input
	os.chdir(c3d_avg_txt_path)
	os.system('cp -r Training_Normal_Videos_Anomaly/ ~/Desktop/AnomalyDetectionCVPR2018/Train_Folder/')
	os.chdir(curr_path)


	#Extracting all Abnormal training input
	f = open("Anomaly_Train.txt", "r") # File containing list input file for training (only Abnormal videos)
	lines = f.readlines()
	for x in lines:
		get_input_for_train = os.path.join(c3d_avg_txt_path, x.replace(".mp4", "_C.txt"))
		os.system("cp {} ~/Desktop/AnomalyDetectionCVPR2018/Train_Folder/Training_Abnormal_Videos_Anomaly/".format(get_input_for_train.rstrip()))
	f.close()

	#Extracting all Testing input
	f_test = open("Anomaly_Test.txt") # File containing list of all testing files
	l_test = f_test.readlines()
	for y in l_test:
		get_input_for_test = os.path.join(c3d_avg_txt_path, y.replace(".mp4", "_C.txt"))
		os.system("cp {} ~/Desktop/AnomalyDetectionCVPR2018/Test_Folder/".format(get_input_for_test.rstrip()))
	f_test.close()


if __name__ == '__main__':
	main()
