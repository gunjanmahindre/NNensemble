 #################################################################

import os
import random
import numpy as np
import pandas as pd
import DeepAE as DAE
import networkx as nx

# path = '/content/drive/MyDrive/PhD work/Projects/ensemble/virgili results/fine tuning/'
# w1 = [4,5,6]
# w2 = [6,7,8]
# w3 = [8,9,10]


def eval_result(path, w1, w2, w3, hadamard_test, fraction, data_path):

	# load original dataset
	# # load data for testing---REAL WORLD NETWORK
	# data_path = '/content/drive/MyDrive/PhD work/data/undirected networks/train bombing/'
	# data_path = '/content/drive/MyDrive/PhD work/data/undirected networks/virgili emails/'
	data_test = np.loadtxt(data_path + 'dHp.txt')
	data_test_original = data_test.copy()


	# ########################### average all the three result matrices ############
	# (matrix 1 + matrix 2 + matrix 3)/3 elementwise

	ss = str(w1[0]) + '_' + str(w1[1]) + '_' + str(w1[2])
	matrix1 = np.loadtxt(path + 'R_' + str(fraction) + '_option_' + ss + '.txt')

	ss = str(w2[0]) + '_' + str(w2[1]) + '_' + str(w2[2])
	matrix2 = np.loadtxt(path + 'R_' + str(fraction) + '_option_' + ss + '.txt')

	ss = str(w3[0]) + '_' + str(w3[1]) + '_' + str(w3[2])
	matrix3 = np.loadtxt(path + 'R_' + str(fraction) + '_option_' + ss + '.txt')

	# take average of the three results
	res = matrix1 + matrix2 + matrix3
	R = np.divide(res,3) 
	R = pd.DataFrame(R)
	print ("-------------------------------------------------------averaged the results------------------------------------------")

	# correct again---------------------

	# # Correction code:
	# """
	# All diagonal entries are set to 0
	# All off diagonal entries predicted <=1 are set to 1
	# """
	for i in range(len(R)):
		for j in R.columns:
			if i != j:
				if R.iloc[i, j] <= 1.0:
					R.iloc[i, j] = 1
				else:
					continue
	for i in range(len(R)):
		for j in R.columns:
			if i == j:
				R.iloc[i, j] = 0
			else:
				continue
	print ("-------------------------------------------------------corrected the results------------------------------------------")



	# ########################## delete the three result matrices #################
	# remove matrix1
	ss = str(w1[0]) + '_' + str(w1[1]) + '_' + str(w1[2])
	file1 = path + '/R_' + str(fraction) + '_option_' + ss + '.txt'
	os.remove(file1)

	# remove matrix2
	ss = str(w2[0]) + '_' + str(w2[1]) + '_' + str(w2[2])
	file2 = path + '/R_' + str(fraction) + '_option_' + ss + '.txt'
	os.remove(file2)

	# remove matrix3
	ss = str(w3[0]) + '_' + str(w3[1]) + '_' + str(w3[2])
	file3 = path + '/R_' + str(fraction) + '_option_' + ss + '.txt'
	os.remove(file3)


	# ########################## evaluate the resultant matrix #####################


	print ("-------------- Calculating error only for unobserved entries--------------------")

	[r,c] = R.shape
	data_test_original = pd.DataFrame(data_test_original)
	# vectorize matrices - placeholders
	hop = []
	ori = []
	# meane = []
	# abse = []


	p = 0
	for i in range(r):
	  for j in range(c):
	    if hadamard_test[i,j] == 0:   # considers error on only unobserved entries
	        hop.append(R.iloc[i,j])
	        ori.append(data_test_original.iloc[i,j])
	    p = p+1

	# #  mean and absolute hop error calculation----------------
	# mean_err: mean error
	# abs_err: AHDE - Absolute hop distance error
	hop = np.array(hop)
	ori = np.array(ori)
	x = np.round(hop-ori)

	# print ("numerator:", np.sum(abs(x)))
	# print ("sum of unobserved entries:", np.sum(ori))
	# print ("b: total unobserved entries:", len(ori))

	mean_err = (np.sum(abs(x)))/(np.sum(ori))        
	mean_err = mean_err*100
	mean_std = np.std(abs(x))

	abs_err = (np.sum(abs(x)))/(len(ori))  # divided by the number of unobserved entries
	abs_std = np.std(abs(x))

	print (mean_err, abs_err, mean_std, abs_std)

	return [mean_err, abs_err, mean_std, abs_std]