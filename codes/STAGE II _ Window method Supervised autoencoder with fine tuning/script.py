"""
AUTOMATION SCRIPT: Supervided Pretrained Autoencoders for Inference in Networks

@author: Gunjan Mahindre
Version: 1.0
Date last modified: Sept. 27 2020

Description: 
    Run the main autoencoder code for various percentages of deletion 
    Run the code over "iter" number of iterations
    Calculates Mean error, Absolute Hop Distance Error (AHDE) averaged over all iterations.

"""

"""
1. create windows
iterate through the windows
2. create these networks.. do 3 Power law
3. train
4. test - only over observed entries - on virgili network

% = 60, 80, 90, 99, 99.5, 99.9

plot : each window as seperate graph.. 
cross check whether the window of actual average node degree performs best..

"""
# IMPORT MODULES REQUIRED

print ('here 0')



import os
import random
import numpy as np
import pandas as pd
# import RobustDeepAutoencoder as auto
from RobustDeepAutoencoder import *
from RobustDeepAutoencoder import RDAE
import DeepAE as DAE
import networkx as nx
from evaluate import eval_result


# # ---------------------------------------------------
# # 1. create windows
# windows = [[5,7,9]]
# print ('here 1')
# w = windows[0]
# print ("current window:---------------------------------------------------------------------------------------- ", w)
# # create Directory for this window----------------
# directory = str(w[0]) + '_' + str(w[1]) + '_' + str(w[2])
# # Parent Directory path  
# parent_dir = "/content/drive/MyDrive/PhD work/Projects/ensemble/virgili results/fine tuning/"

# # Path  
# path = os.path.join(parent_dir, directory)   
# Create the directory   
# os.mkdir(path)  
# print("Directory '% s' created" % directory) 
# # ---------------------------------------------------

path = '/content/drive/MyDrive/PhD work/Projects/ensemble/virgili results/fine tuning/'
# path = '/content/drive/MyDrive/PhD work/Projects/ensemble/train bombing results/fine tuning/'

# data_path = '/content/drive/MyDrive/PhD work/data/undirected networks/train bombing/'
data_path = '/content/drive/MyDrive/PhD work/data/undirected networks/virgili emails/'

print ("RESULTS FOR SUPERVISED AUTOENCODERS ")

mean_results = []
abs_results = []
m_STD_results = []
a_STD_results = []

frac_list = [20, 40, 60, 80, 90, 99, 99.5, 99.9]
# frac_list = [20]

print ('here 2')

for fraction in frac_list:
	# main_code(fraction, w)
	print ("Fraction--------------------------------", fraction)

	# for the given fraction----

	# run option 1
	# window for this option
	w1 = [4,5,6]
	print ("current window:---------------------------------------------------------------------------------------- ", w1)
	hadamard_test1 = main_code(fraction, w1, path, data_path)
	hadamard_test1 = np.array(hadamard_test1)
	# save the corrected result matrix 

	print('done with option 1')

	# # run option 2
	# # window for this option
	w2 = [6,7,8]
	print ("current window:---------------------------------------------------------------------------------------- ", w2)
	hadamard_test2 = main_code(fraction, w2, path, data_path)
	hadamard_test2 = np.array(hadamard_test2)
	# # save the corrected result matrix

	print('done with option 2')

	# # check if the same entries are being deleted.. so that we can average these entries===they ARE same :)
	# if np.sum(hadamard_test1 - hadamard_test2) == 0:
	# 	print ('same')

	# # run option 3
	# # window for this option
	w3 = [8,9,10]
	print ("current window:---------------------------------------------------------------------------------------- ", w3)
	hadamard_test3 = main_code(fraction, w3, path, data_path)
	hadamard_test3 = np.array(hadamard_test3)
	# # save the corrected result matrix

	print('done with option 3')

	# average the three result matrices 
	# evaluate this final result
	[mean_err, abs_err, mean_std, abs_std] = eval_result(path, w1, w2, w3, hadamard_test1, fraction, data_path)
	print(mean_err, abs_err, mean_std, abs_std)

# 	# append the result to our variables :)

	mean_results.append(mean_err)
	abs_results.append(abs_err)
	m_STD_results.append(mean_std)
	a_STD_results.append(abs_std)


# save each result in a text file
filename = '/mean_error.txt'
np.savetxt(path + filename, mean_results)
filename = '/abs_error.txt'
np.savetxt(path + filename, abs_results)
filename = '/mean_STD.txt'
np.savetxt(path + filename, m_STD_results)
filename = '/abs_STD.txt'
np.savetxt(path + filename, a_STD_results)

print (frac_list)

exit()