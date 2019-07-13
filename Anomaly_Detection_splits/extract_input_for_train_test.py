import os

def main():
	c3d_avg_txt_path = "~/Desktop/AnomalyDetectionCVPR2018/C3D_features_txt_Avg"
	f = open("Anomaly_Train.txt", "r") # File containing list input file for training (only Abnormal videos)

	lines = f.readlines()
	for x in lines:
		get_input_for_train = os.path.join(c3d_avg_txt_path, x.replace(".mp4", "_C.txt"))
		os.system("cp {} ~/Desktop/AnomalyDetectionCVPR2018/Train_Folder/Training_Abnormal_Videos_Anomaly/".format(get_input_for_train.rstrip()))	
	
	f.close()


	f_test = open("Anomaly_Test.txt") # File containing list of all testing files
	l_test = f_test.readlines()
	for y in l_test:
		get_input_for_test = os.path.join(c3d_avg_txt_path, y.replace(".mp4", "_C.txt"))
		os.system("cp {} ~/Desktop/AnomalyDetectionCVPR2018/Test_Folder/".format(get_input_for_test.rstrip()))	

	f_test.close()


if __name__ == '__main__':
	main()
